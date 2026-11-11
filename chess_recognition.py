"""Offline chessboard detection and piece recognition for ChessSnip.

The square classifier is the pretrained MIT-licensed chessimg2pos model,
exported to ONNX for a compact CPU-only runtime. Board-line detection is
adapted from mdicio/chessimg2pos' MIT-licensed chessboard_finder.py.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

FEN_CHARS = "1RNBQKPrnbqkp"
MAX_DETECT_DIMENSION = 1500


class ChessRecognitionError(Exception):
    """Raised when a board cannot be detected or classified."""


def _all_sequences(values: np.ndarray, min_len: int = 7, tolerance: int = 5):
    sequences = []
    for i in range(len(values) - 1):
        for j in range(i + 1, len(values)):
            spacing = int(values[j] - values[i])
            if spacing < tolerance:
                continue
            sequence = [int(values[i]), int(values[j])]
            target = sequence[-1] + spacing
            while len(sequence) < 10 and len(values):
                distances = np.abs(values - target)
                nearest = int(distances.argmin())
                if distances[nearest] >= tolerance:
                    break
                found = int(values[nearest])
                if found <= sequence[-1]:
                    break
                sequence.append(found)
                target = found + spacing
            if len(sequence) < min_len:
                continue
            key = tuple(sequence)
            if any(len(old) >= 2 and old[:2] == key[:2] for old in sequences):
                continue
            sequences.append(key)
    return [np.asarray(sequence, dtype=int) for sequence in sequences]


def _suppress_peaks(values: np.ndarray, window: int = 5):
    peaks = values.copy()
    for index in range(values.size):
        left = values[max(0, index - window):index]
        right = values[index + 1:min(values.size, index + window + 1)]
        if (left.size and values[index] < left.max()) or (right.size and values[index] <= right.max()):
            peaks[index] = 0
    return peaks


def _best_line_sequence(hough: np.ndarray):
    peaks = _suppress_peaks(hough)
    maximum = float(peaks.max(initial=0))
    if maximum <= 0:
        return None
    peaks = peaks / maximum
    peaks[peaks < 0.2] = 0
    positions = np.flatnonzero(peaks)
    sequences = _all_sequences(positions)
    if not sequences:
        return None
    scored = []
    for sequence in sequences:
        strengths = peaks[sequence]
        while len(sequence) > 9:
            if strengths[0] > strengths[-1]:
                sequence, strengths = sequence[:-1], strengths[:-1]
            else:
                sequence, strengths = sequence[1:], strengths[1:]
        scored.append((float(strengths.mean()), sequence))
    return max(scored, key=lambda item: item[0])[1]


def _detect_board_bounds(gray: np.ndarray):
    """Find a level, screen-rendered 8x8 checkerboard in grayscale pixels.

    Returns bounds in the same order as the source detector: (top, left,
    bottom, right), or None if a regular 7+ line grid is not visible.
    """
    gx, gy = np.gradient(gray.astype(np.float64, copy=False))
    gx_pos, gx_neg = np.maximum(gx, 0), np.maximum(-gx, 0)
    gy_pos, gy_neg = np.maximum(gy, 0), np.maximum(-gy, 0)
    hough_y = gx_pos.sum(axis=1) * gx_neg.sum(axis=1)
    hough_x = gy_pos.sum(axis=0) * gy_neg.sum(axis=0)
    if min(hough_y.std() / max(1, hough_y.size), hough_x.std() / max(1, hough_x.size)) < 8000:
        return None

    x_lines = _best_line_sequence(hough_x)
    y_lines = _best_line_sequence(hough_y)
    if x_lines is None or y_lines is None:
        return None

    dx = float(np.median(np.diff(x_lines)))
    dy = float(np.median(np.diff(y_lines)))
    if dx <= 0 or dy <= 0:
        return None
    x_windows = [x_lines[i:i + 7] for i in range(max(1, len(x_lines) - 6))]
    y_windows = [y_lines[i:i + 7] for i in range(max(1, len(y_lines) - 6))]
    base_top, base_left = int(y_lines[0] - dy), int(x_lines[0] - dx)
    base_bottom, base_right = int(y_lines[-1] + dy), int(x_lines[-1] + dx)
    gray_image = Image.fromarray(gray.astype(np.uint8, copy=False))
    square = np.ones((8, 8), dtype=np.float64)
    kernel = np.tile(np.block([[square, -square], [-square, square]]), (4, 4))
    kernel /= np.linalg.norm(kernel)
    best_bounds, best_score = None, -1.0
    for y_window in y_windows:
        for x_window in x_windows:
            top = int(y_window[0] - dy)
            left = int(x_window[0] - dx)
            bottom = int(y_window[-1] + dy)
            right = int(x_window[-1] + dx)
            relative = (left - base_left, top - base_top, right - base_left, bottom - base_top)
            candidate = gray_image.crop(relative).resize((64, 64), Image.Resampling.BILINEAR)
            score = abs(float(np.sum(kernel * np.asarray(candidate, dtype=np.float64))))
            if score > best_score:
                best_score, best_bounds = score, (top, left, bottom, right)
    return best_bounds


def _board_from_screenshot(image: Image.Image, refine: bool = True):
    original_w, original_h = image.size
    scale = min(1.0, MAX_DETECT_DIMENSION / max(original_w, original_h))
    working = image.convert("L")
    if scale < 1.0:
        working = working.resize((max(1, round(original_w * scale)), max(1, round(original_h * scale))), Image.Resampling.BILINEAR)
    gray = np.asarray(working, dtype=np.uint8)
    bounds = _detect_board_bounds(gray)
    if bounds is None:
        raise ChessRecognitionError("I couldn't automatically locate a chessboard in this snip.")
    top, left, bottom, right = bounds
    factor = 1.0 / scale
    left, top = round(left * factor), round(top * factor)
    right, bottom = round(right * factor), round(bottom * factor)
    left, top = max(0, left), max(0, top)
    right, bottom = min(original_w, right), min(original_h, bottom)
    width, height = right - left, bottom - top
    if min(width, height) < 80 or abs(width - height) / max(width, height) > 0.12:
        raise ChessRecognitionError("I couldn't isolate a square chessboard. Select just the board and try again.")
    if refine and scale < 1.0:
        # Bounds found on a downscaled image are off by a few pixels, which
        # shifts every tile. Re-detect at full resolution around the board.
        pad = round(max(width, height) * 0.06)
        region = image.crop((
            max(0, left - pad), max(0, top - pad),
            min(original_w, right + pad), min(original_h, bottom + pad),
        ))
        try:
            return _board_from_screenshot(region, refine=False)
        except ChessRecognitionError:
            pass
    return image.crop((left, top, right, bottom))


def _piece_orientation_score(rows):
    """Estimate which color is at the bottom from the detected piece layout.

    Positive scores favor White at the bottom. The board's square colors cannot
    distinguish a 180-degree rotation, so use the usual piece and pawn layout.
    """
    score = total_weight = 0.0
    for row_index, row in enumerate(rows):
        vertical_position = (row_index - 3.5) / 3.5
        for piece in row:
            if piece == "1":
                continue
            kind = piece.upper()
            weight = 5.0 if kind == "K" else 2.5 if kind == "P" else 0.5
            color_direction = 1.0 if piece.isupper() else -1.0
            score += vertical_position * color_direction * weight
            total_weight += weight
    return score / total_weight if total_weight else 0.0


@lru_cache(maxsize=1)
def _session():
    if getattr(sys, "frozen", False):
        model_path = Path(sys._MEIPASS) / "chess_position_classifier.onnx"
    else:
        model_path = Path(__file__).resolve().with_name("chess_position_classifier.onnx")
    if not model_path.exists():
        raise ChessRecognitionError("The offline chess recognition model is missing. Reinstall ChessSnip to restore it.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def _classify_board(board: Image.Image):
    board = board.resize((256, 256), Image.Resampling.BILINEAR)
    gray = np.asarray(board.convert("L", (0.2989, 0.5870, 0.1140, 0)), dtype=np.float32)
    tiles = gray.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 1, 32, 32)
    batch = (tiles / 127.5 - 1.0).astype(np.float32, copy=False)
    logits = _session().run(["piece_logits"], {"tile": batch})[0]
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    rows = ["".join(FEN_CHARS[int(labels[row * 8 + col])] for col in range(8)) for row in range(8)]
    return board, rows, confidences


def recognize_position(image: Image.Image, board_crop: bool = False):
    """Recognize a screenshot, normalize its orientation, and return FEN rows."""
    image = image.convert("RGB")
    board = image if board_crop else _board_from_screenshot(image)
    source_board = board.copy()
    board, rows, confidences = _classify_board(board)
    orientation_score = _piece_orientation_score(rows)
    orientation_flipped = orientation_score < -0.08
    if orientation_flipped:
        # Classify the original artwork only once. Rotating piece glyphs before
        # the second pass can change a pawn into another piece class. Rotate the
        # already recognized squares instead, preserving every piece label.
        board = board.rotate(180)
        rows = [row[::-1] for row in reversed(rows)]
        confidences = confidences.reshape(8, 8)[::-1, ::-1].reshape(64)
    return {
        # Keep the detected source image untouched. Analysis rows are normalized
        # separately so the review window can rotate only the piece preview.
        "board": source_board,
        "rows": rows,
        "confidence": float(confidences.mean()),
        "tile_confidence": confidences.reshape(8, 8),
        "orientation_flipped": orientation_flipped,
        "orientation_uncertain": abs(orientation_score) <= 0.08,
    }


