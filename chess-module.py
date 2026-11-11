"""Chess block: find a board on the screenshot, ask Stockfish, return text for Yogi.

Workflow:
1. Recognize the board (line detection, then a whole-snip fallback for tight snips).
2. Build a legal position for each side to move, with castling rights inferred
   from piece placement. A side whose opponent's king is in check cannot be
   the side to move, which tells us whose turn it is.
3. Ask Stockfish for the top lines per legal side, in parallel.
4. Return a fixed-format block with the best line as a numbered list of
   NUM_MOVES moves per side, so the model reads them in order. Tone and
   length come from Yogi's main prompt, not from this module.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import chess
import chess.engine
from chess_recognition import ChessRecognitionError, recognize_position

_MODULE_DIR = Path(__file__).resolve().parent
STOCKFISH_PATH = str(
    _MODULE_DIR / "stockfish19" / "stockfish-windows-x86-64-universal.exe"
)
THINK_TIME = 1.0   # seconds per side
NUM_LINES = 1      # best move + one alternative per side
NUM_MOVES = 3      # moves per side in the line Yogi reads out

_MIN_CROP_CONFIDENCE = 0.45   # whole-snip fallback must be at least this sure
_LOW_TILE_CONFIDENCE = 0.80   # warn the AI if any single square is below this

# Sent with Yogi's system prompt when a board is found, followed by the analysis.
PROMPT = """
### CHESS MODULE - MOVE RULES
The CHESS ANALYSIS below comes from a chess engine reading the board on the user's screen. It is the only source of chess moves. These rules decide WHICH moves you say. How you say them (tone, personality, length, extra commentary) follows your main instructions above.
1. Only say moves that are written in the analysis. Never invent, change, or calculate a move yourself, and never read moves off the image.
2. Which side: if the user names White or Black, answer only for that side. If not, and "Side to move" names a side, answer for that side. Otherwise answer for White, then Black.
3. For each side you answer, say every move of its "Best line" in the numbered order, and say whose move each one is. Move 1 is the best move. Never skip, reorder, or add moves.
4. Say each move as written, for example "rook from e5 to e7". You may connect them naturally ("then", "after that"), but keep squares and pieces exact.
5. Mention the evaluation or the alternative first move only if the user asks, or if your main instructions make you more talkative.
6. If the analysis has a "Warning" line, mention once that the board might be misread.
7. If the user asks about a side that has no section, say that side cannot be to move in this position.
8. If the user asks you to repeat, repeat the same line from the current analysis.
""".strip()

PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}
SIDE_NAMES = {chess.WHITE: "White", chess.BLACK: "Black"}

# Last analysis by piece placement, so an unchanged board on a follow-up
# skips Stockfish.
_cache_lock = threading.Lock()
_cache = {"placement": None, "text": ""}


def rows_to_fen(rows):
    out = []
    for row in rows:
        s, count = "", 0
        for ch in row:
            if ch == "1":
                count += 1
            else:
                if count:
                    s += str(count)
                    count = 0
                s += ch
        if count:
            s += str(count)
        out.append(s)
    return "/".join(out)


def move_to_words(board, move):
    """Describe one move in words, then play it on the board."""
    if board.is_kingside_castling(move):
        text = "castles kingside"
    elif board.is_queenside_castling(move):
        text = "castles queenside"
    else:
        piece = PIECE_NAMES[board.piece_type_at(move.from_square)]
        source = chess.square_name(move.from_square)
        target = chess.square_name(move.to_square)
        if board.is_en_passant(move):
            text = f"{piece} from {source} takes pawn on {target} en passant"
        elif board.is_capture(move):
            captured = PIECE_NAMES[board.piece_type_at(move.to_square)]
            text = f"{piece} from {source} takes {captured} on {target}"
        else:
            text = f"{piece} from {source} to {target}"
        if move.promotion:
            text += f", promoting to {PIECE_NAMES[move.promotion]}"
    board.push(move)
    if board.is_checkmate():
        text += ", checkmate"
    elif board.is_check():
        text += ", check"
    return text


def score_to_words(score):
    white = score.white()
    if white.is_mate():
        n = white.mate()
        if n == 0:
            return "checkmate on the board"
        who = "White" if n > 0 else "Black"
        return f"{who} has a forced mate in {abs(n)}"
    cp = white.score()
    if abs(cp) < 50:
        return "the position is roughly equal"
    who = "White" if cp > 0 else "Black"
    return f"{who} is better by about {abs(cp) / 100:.1f} pawns"


def _position_for(placement, color):
    """Board with `color` to move and castling rights the placement allows."""
    side = "w" if color == chess.WHITE else "b"
    board = chess.Board(f"{placement} {side} KQkq - 0 1")
    board.castling_rights = board.clean_castling_rights()
    return board


def _legal_positions(placement):
    """{color: board} for every side that can legally be to move."""
    boards = {}
    for color in (chess.WHITE, chess.BLACK):
        board = _position_for(placement, color)
        if board.is_valid():
            boards[color] = board
        else:
            print(
                f"chess: {SIDE_NAMES[color]} to move is not legal "
                f"({chess.Status(board.status())!r})",
                flush=True,
            )
    return boards


def _analyse(board):
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        infos = engine.analyse(
            board, chess.engine.Limit(time=THINK_TIME), multipv=NUM_LINES
        )
    return [i for i in infos if i.get("pv")]


def _side_section(board, infos):
    """Fixed-format block for one side. Best move is always item 1."""
    name = SIDE_NAMES[board.turn]
    if board.is_checkmate():
        return [f"{name.upper()} TO MOVE:", f"  {name} is checkmated. No moves."]
    if board.is_stalemate():
        return [f"{name.upper()} TO MOVE:", "  Stalemate. No moves."]
    if not infos:
        return []

    best = infos[0]
    b = board.copy()
    lines = [f"{name.upper()} TO MOVE:", "  Best line (say every move, in order):"]
    for i, move in enumerate(best["pv"][: NUM_MOVES * 2], 1):
        lines.append(f"    {i}. {SIDE_NAMES[b.turn]} {move_to_words(b, move)}")
    lines.append(f"  Evaluation: {score_to_words(best['score'])}.")
    if len(infos) > 1:
        alt = infos[1]
        lines.append(
            f"  Alternative first move: {move_to_words(board.copy(), alt['pv'][0])} "
            f"({score_to_words(alt['score'])})."
        )
    return lines


def _side_to_move_line(boards):
    if len(boards) == 2:
        return "Side to move: unknown, both sides analysed."
    (color,) = boards
    other = SIDE_NAMES[not color]
    return (
        f"Side to move: {SIDE_NAMES[color]} "
        f"({other} cannot be to move here, {SIDE_NAMES[color]}'s king is in check)."
    )


def _low_confidence_squares(tile_confidence):
    squares = []
    for row in range(8):
        for col in range(8):
            if tile_confidence[row][col] < _LOW_TILE_CONFIDENCE:
                squares.append(chess.square_name(chess.square(col, 7 - row)))
    return squares


def _looks_like_board_snip(image):
    """Whole-image-as-board is only safe on a tight snip, not a desktop."""
    try:
        width, height = image.size
    except Exception:
        return False
    if min(width, height) < 80 or max(width, height) > 1200:
        return False
    return abs(width - height) / max(width, height) <= 0.25


def _recognize(image):
    """Find the board. If line-detect fails, try treating a small snip as the board."""
    try:
        return recognize_position(image)
    except ChessRecognitionError as first:
        print(f"chess: {first}", flush=True)
        if not _looks_like_board_snip(image):
            return None
        try:
            print("chess: retrying with the whole snip as the board", flush=True)
            result = recognize_position(image, board_crop=True)
        except Exception as exc:
            print(f"chess: {exc}", flush=True)
            return None
        if result["confidence"] < _MIN_CROP_CONFIDENCE:
            print(
                f"chess: whole-snip confidence {result['confidence']:.2f} "
                f"below {_MIN_CROP_CONFIDENCE}; skipping",
                flush=True,
            )
            return None
        return result
    except Exception as exc:
        print(f"chess: recognition failed: {exc}", flush=True)
        return None


def run(image) -> str:
    """Return prompt + Stockfish analysis for Yogi, or "" if no board / error."""
    if not Path(STOCKFISH_PATH).is_file():
        print(f"chess: Stockfish not found at {STOCKFISH_PATH}", flush=True)
        return ""

    result = _recognize(image)
    if result is None:
        return ""

    placement = rows_to_fen(result["rows"])
    print(
        f"chess: board found (confidence {result['confidence']:.2f}): {placement}",
        flush=True,
    )

    boards = _legal_positions(placement)
    if not boards:
        print("chess: position is not legal for either side; skipping", flush=True)
        return ""

    with _cache_lock:
        if _cache["placement"] == placement:
            print("chess: board unchanged, reusing last analysis", flush=True)
            return _cache["text"]

    print("chess: asking Stockfish...", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=len(boards)) as pool:
            infos = dict(zip(boards, pool.map(_analyse, boards.values())))
    except Exception as exc:
        print(f"chess: stockfish failed: {exc}", flush=True)
        return ""

    sections = []
    for color, board in boards.items():
        lines = _side_section(board, infos[color])
        if lines:
            sections.append("\n".join(lines))
    if not sections:
        print("chess: Stockfish returned no lines; skipping", flush=True)
        return ""

    header = [
        "CHESS ANALYSIS",
        f"Board (FEN, piece placement only): {placement}",
        f"Board read confidence: {result['confidence']:.2f}",
        _side_to_move_line(boards),
    ]
    weak = _low_confidence_squares(result["tile_confidence"])
    if weak:
        header.append(
            f"Warning: these squares were hard to read: {', '.join(weak)}."
        )
    analysis = "\n".join(header) + "\n\n" + "\n\n".join(sections)
    print(f"chess: ready\n{analysis}", flush=True)

    text = "\n\n".join(part for part in (PROMPT, analysis) if part)
    with _cache_lock:
        _cache["placement"] = placement
        _cache["text"] = text
    return text


if __name__ == "__main__":
    import sys
    from PIL import Image

    if len(sys.argv) < 2:
        print("usage: python chess-module.py <board.png>", file=sys.stderr)
        raise SystemExit(1)
    text = run(Image.open(sys.argv[1]))
    print(text or "(no board / no lines)")
