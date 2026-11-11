# Yogi Chess

Yogi block that finds a chessboard on a screenshot, reads the pieces offline, asks Stockfish for the best lines, and returns text Yogi's vision model must follow when talking about moves.

## What it does

On each screenshot Yogi sends to this block:

1. Locate an 8x8 board in the image (line detection). If that fails and the image looks like a tight board snip, treat the whole image as the board.
2. Classify each square with an ONNX piece model, normalize orientation (White at the bottom when the layout is clear), and build FEN piece placement.
3. Keep only legal side-to-move positions. If one side's king is in check, the other side cannot be to move.
4. Run Stockfish (about 1s per legal side, in parallel) and format a fixed analysis block: best line as numbered word moves, evaluation, optional low-confidence square warning.
5. Prefix that analysis with move rules for the model: only say moves from the analysis, keep order, use the written squares and pieces.

If no board is found, the position is illegal for both sides, Stockfish is missing, or analysis fails, the block prints nothing useful for Yogi (empty result). An unchanged board reuses the last analysis.

## Install in Yogi

1. Put this folder under Yogi's `modules/` directory, for example `modules/chess/`, or install a zip from Yogi's Blocks tab.
2. Folder / zip id must match Yogi's rules: start with a letter, then letters, digits, or underscores (max 32 chars, lowercase).
3. Enable the block in the Blocks tab (or set `"enabled": true` in `manifest.json`).
4. Keep the bundled Stockfish binary at `stockfish19/stockfish-windows-x86-64-universal.exe` relative to this folder.

Yogi runs:

```text
python chess-module.py <temp.png>
```

Working directory is the block folder. Result text goes to stdout; empty stdout means "nothing to add". Logs should go to stderr. Timeout is short (about 8s in Yogi), so keep Stockfish think time modest.

## Layout

| Path | Role |
|------|------|
| `manifest.json` | Block name, description, entry (`chess-module.py`) |
| `chess-module.py` | Orchestration, Stockfish, output for Yogi |
| `chess_recognition.py` | Board find + square classification |
| `chess_position_classifier.onnx` | Offline piece classifier (CPU ONNX Runtime) |
| `stockfish19/` | Stockfish 19 sources, license, Windows universal binary |
| `requirements.txt` | Python deps for a local dry run |
| `LICENSE.txt` | Project license (GPL-3.0-or-later) |
| `THIRD_PARTY_NOTICES.txt` | ChessSnip, chessimg2pos, Stockfish, python-chess notices |

## Local dry run

From this folder, with dependencies installed (`pip install -r requirements.txt`):

```bash
python chess-module.py path\to\board.png
```

That prints the prompt + analysis (or a "no board" message). Use any screenshot or snip of a chessboard.

Python 3.10+ on Windows is the target (matches the bundled `.exe`).

## How Yogi should use the output

The block returns two parts joined together:

1. **Move rules** - tell the model to recite only the engine lines, pick a side from the user or from "Side to move", and mention recognition warnings once when present.
2. **CHESS ANALYSIS** - FEN placement, confidence, side-to-move hint, then per-side "Best line" moves in plain English (`rook from e5 to e7`, castling, captures, mate).

Tone and length stay in Yogi's main system prompt. This block only supplies legal moves and the rules for which ones to say.

## Knobs (in `chess-module.py`)

These are editable constants near the top of `chess-module.py`:

- `PROMPT` - move rules prepended to every analysis. Edit this when you want Yogi to follow different rules for which moves to say, how to pick a side, or when to mention warnings / evaluations. Tone and personality still come from Yogi's main system prompt.
- `THINK_TIME` - how many seconds Stockfish thinks per legal side (default `1.0`). Higher = stronger / slower analysis; keep it low so the block stays under Yogi's ~8s timeout.
- `NUM_LINES` - how many principal variations Stockfish returns per side (default `1`). Set to `2+` if you want an "Alternative first move" line in the analysis.
- `NUM_MOVES` - how many half-moves from the best line are written out for Yogi to recite per side (default `3`). The code walks `NUM_MOVES * 2` plies so both colors appear in the spoken line.
- `_MIN_CROP_CONFIDENCE` - minimum overall piece-read confidence (default `0.45`) required when line detection failed and the block retries by treating a tight snip as the whole board. Below this, the snip is skipped.
- `_LOW_TILE_CONFIDENCE` - per-square confidence threshold (default `0.80`). Any square below this is listed in a "Warning: these squares were hard to read" line so the model can mention the board might be misread.

## Credits and licenses

This block is **GPL-3.0-or-later** — see `LICENSE.txt`. Full third-party notices are in `THIRD_PARTY_NOTICES.txt`.

- Board recognition (`chess_recognition.py` + ONNX model) is adapted from [ChessSnip](https://github.com/DaoraDio/ChessSnip) (MIT), which itself uses the [chessimg2pos](https://github.com/mdicio/chessimg2pos) finder and piece model (MIT).
- Move legality and UCI glue use [python-chess](https://github.com/niklasf/python-chess) (GPL-3.0-or-later).
- Engine analysis uses [Stockfish](https://stockfishchess.org/) under GPL v3 — see `stockfish19/Copying.txt` and `stockfish19/README.md`. The bundled binary ships with corresponding source under `stockfish19/src/` as GPL requires.
