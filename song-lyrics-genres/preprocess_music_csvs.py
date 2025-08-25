#!/usr/bin/env python3
"""
Preprocess music CSVs: normalize genres and lightly clean lyrics (no tokenization).

- Reads all CSVs matching a glob under an input directory.
- Detects the likely "genres" and "lyrics" columns (case-insensitive, flexible).
- Normalizes the genres column so each row is a comma-separated list (e.g., "pop, dance pop").
  * Accepts existing Python-list literals, JSON-like arrays, or strings separated by commas,
    semicolons, slashes, pipes, or " | " etc.
  * Lowercases, strips, collapses internal whitespace, deduplicates while preserving order.
- Lightly cleans the lyrics column without tokenization:
  * Unicode normalization (NFKC)
  * Normalizes line endings to "\n"
  * Collapses runs of blank lines to a single blank line
  * Trims trailing spaces on each line and strips leading/trailing whitespace
- Writes out to <original_name>_preprocessed.csv by default, or to an output directory.
- Prints a simple per-file summary of changes.

Usage:
    python preprocess_music_csvs.py --in /path/to/csvs --glob "*lyrics*.csv" --out /path/to/outdir

Tip:
    If your CSVs have unusual encodings, pass e.g. --encoding "utf-8-sig" or --encoding "latin-1".
"""
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import re
import sys
import unicodedata
import ast
from typing import List, Optional, Tuple, Iterable, Union, Any, Dict

# ----------------------------
# Column detection helpers
# ----------------------------

CANDIDATE_GENRE_COLS = ["genres", "genre", "tags"]
CANDIDATE_LYRICS_COLS = ["lyrics", "lyric", "text", "body"]

def detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the best-matching column name from candidates (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        # direct match
        if name in lower_map:
            return lower_map[name]
        # case-insensitive contains (e.g., 'song_lyrics' matches 'lyrics')
        for lc, orig in lower_map.items():
            if name in lc:
                return orig
    return None

# ----------------------------
# Genre normalization
# ----------------------------

_SPLIT_REGEX = re.compile(r"[;,/|]|\s+\&\s+|\s+\+\s+|\s+and\s+", flags=re.IGNORECASE)

def parse_genre_cell(val: Any) -> List[str]:
    """Normalize a genre cell into a list of strings."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        raw = val
    else:
        s = str(val).strip()
        if not s or s.lower() in {"nan", "none"}:
            return []
        # Try Python/JSON-like literal
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    raw = list(parsed)
                else:
                    raw = [str(parsed)]
            except Exception:
                raw = _SPLIT_REGEX.split(s)
        else:
            raw = _SPLIT_REGEX.split(s)

    # Clean each token
    cleaned: List[str] = []
    seen = set()
    for tok in raw:
        if tok is None or (isinstance(tok, float) and np.isnan(tok)):
            continue
        t = str(tok).strip().lower()
        t = re.sub(r"\s+", " ", t)  # collapse spaces
        # Remove surrounding quotes if present
        t = t.strip("\"'")
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            cleaned.append(t)
    return cleaned

def genres_to_csv_string(genres: List[str]) -> str:
    """Join genre list into a single comma+space separated string."""
    return ", ".join(genres)

# ----------------------------
# Lyrics cleaning
# ----------------------------

def clean_lyrics(val: Any) -> Any:
    """Lightly clean lyrics without tokenization; preserve as a single string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return val
    s = str(val)
    # Unicode normalization
    s = unicodedata.normalize("NFKC", s)
    # Normalize line endings to \n
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces on each line
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    # Collapse runs of blank lines to a single blank line
    s = re.sub(r"\n{3,}", "\n\n", s)
    # Trim overall
    s = s.strip()
    return s

# ----------------------------
# Main processing
# ----------------------------

def process_file(
    path: Path,
    outdir: Optional[Path],
    encoding: str = "utf-8",
    lyrics_col_hint: Optional[str] = None,
    genre_col_hint: Optional[str] = None,
    keep_empty_genres: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """Process a single CSV and return output path and stats."""
    df = pd.read_csv(path, encoding=encoding)
    orig_len = len(df)

    # Detect columns
    genre_col = genre_col_hint or detect_column(df, CANDIDATE_GENRE_COLS)
    lyrics_col = lyrics_col_hint or detect_column(df, CANDIDATE_LYRICS_COLS)

    if genre_col is None:
        print(f"[WARN] No genre-like column found in {path.name}; skipping genre normalization.", file=sys.stderr)
    else:
        # Normalize genres to a comma-separated string
        parsed = df[genre_col].apply(parse_genre_cell)
        df[genre_col] = parsed.apply(genres_to_csv_string)
        if not keep_empty_genres:
            df = df[df[genre_col] != ""]

    if lyrics_col is None:
        print(f"[WARN] No lyrics-like column found in {path.name}; skipping lyrics cleaning.", file=sys.stderr)
    else:
        df[lyrics_col] = df[lyrics_col].apply(clean_lyrics)

    # Prepare output path
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / f"{path.stem}_preprocessed.csv"
    else:
        out_path = path.with_name(f"{path.stem}_preprocessed.csv")

    df.to_csv(out_path, index=False, encoding=encoding)

    stats = {
        "input_rows": orig_len,
        "output_rows": len(df),
        "genre_col": genre_col,
        "lyrics_col": lyrics_col,
        "output_path": str(out_path),
    }
    return out_path, stats

def process_dir(
    indir: Path,
    glob: str = "*.csv",
    outdir: Optional[Path] = None,
    encoding: str = "utf-8",
    lyrics_col_hint: Optional[str] = None,
    genre_col_hint: Optional[str] = None,
    keep_empty_genres: bool = True,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    paths = sorted(indir.glob(glob))
    if not paths:
        print(f"[WARN] No files matched {glob} under {indir}", file=sys.stderr)
        return results

    for p in paths:
        try:
            out_path, stats = process_file(
                p, outdir, encoding, lyrics_col_hint, genre_col_hint, keep_empty_genres
            )
            print(f"[OK] {p.name} -> {out_path.name} | genres='{stats['genre_col']}' lyrics='{stats['lyrics_col']}'")
            results.append(stats)
        except Exception as e:
            print(f"[ERROR] Failed to process {p}: {e}", file=sys.stderr)
    return results

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Preprocess music CSVs (genres + lyrics).")
    ap.add_argument("--in", dest="indir", required=True, type=Path, help="Input directory containing CSVs")
    ap.add_argument("--glob", dest="glob", default="*.csv", help="Glob for CSV selection (default: *.csv)")
    ap.add_argument("--out", dest="outdir", type=Path, default=None, help="Output directory (default: alongside inputs)")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding for read/write (default: utf-8)")
    ap.add_argument("--lyrics-col", dest="lyrics_col", default=None, help="Explicit lyrics column name (optional)")
    ap.add_argument("--genre-col", dest="genre_col", default=None, help="Explicit genre column name (optional)")
    ap.add_argument("--drop-empty-genres", action="store_true", help="Drop rows whose genres normalize to empty")
    return ap

def main(argv: Optional[List[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    results = process_dir(
        indir=args.indir,
        glob=args.glob,
        outdir=args.outdir,
        encoding=args.encoding,
        lyrics_col_hint=args.lyrics_col,
        genre_col_hint=args.genre_col,
        keep_empty_genres=not args.drop_empty_genres,
    )

    if not results:
        return 1

    # Print a simple summary footer
    total_in = sum(r["input_rows"] for r in results)
    total_out = sum(r["output_rows"] for r in results)
    files = len(results)
    print(f"\nProcessed {files} file(s). Total rows: {total_in} -> {total_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
