#!/usr/bin/env python3
"""
Fill missing 'genres' values in 2015–2024 song_lyrics_and_genres.csv files
using Spotify artist profile genres (main artist only).

Usage:
  1) Install dependencies:
       pip install -r requirements.txt
  2) Provide Spotify API credentials via env vars OR CLI flags:
       export SPOTIFY_CLIENT_ID="..."
       export SPOTIFY_CLIENT_SECRET="..."
     or
       python fill_spotify_genres.py --client-id ... --client-secret ...
  3) Run:
       python fill_spotify_genres.py --data-dir /path/to/csvs

Output:
  - Writes updated CSVs alongside originals with suffix *_genres_filled.csv
  - Writes a cache file spotify_artist_genres_cache.json
  - Writes a run report genres_fill_report.csv
"""

import os
import re
import json
import time
import argparse
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

YEARS = list(range(2015, 2025))
CACHE_FILE = "spotify_artist_genres_cache.json"
REPORT_FILE = "genres_fill_report.csv"

FEAT_SPLIT = re.compile(r"\s+(?:feat\.?|featuring|with)\s+", re.IGNORECASE)
COLLAB_SPLIT = re.compile(r"\s*(?:,|&|and| x | X |;|/|\+)\s*")


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Heuristically detect artist/title/genre columns."""
    artist_cols = [c for c in df.columns if c.lower() in {"artist(s)", "artists", "main_artist", "primary_artist"}]
    title_cols = [c for c in df.columns if c.lower() in {"title", "song", "track", "track_name", "song_title"}]
    genre_cols = [c for c in df.columns if "genre" in c.lower()]
    artist_col = artist_cols[0] if artist_cols else None
    title_col = title_cols[0] if title_cols else None
    genre_col = None
    for c in df.columns:
        if c.lower() == "genres":
            genre_col = c
            break
    if genre_col is None and genre_cols:
        genre_col = genre_cols[0]
    return artist_col, title_col, genre_col


def extract_main_artist(artist_str: str) -> Optional[str]:
    """Extract main artist by removing 'feat.' parts and taking the first primary name."""
    if artist_str is None or not isinstance(artist_str, str):
        return None
    s = artist_str.strip()
    s = FEAT_SPLIT.split(s)[0]
    parts = COLLAB_SPLIT.split(s)
    return parts[0].strip() if parts else s.strip()


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation/accents for basic matching."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # keep alphanum and spaces only
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class SpotifyClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self._token = None
        self._token_expiry = 0

    def _get_token(self) -> str:
        now = time.time()
        if self._token and now < self._token_expiry - 30:
            return self._token
        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = now + int(data.get("expires_in", 3600))
        return self._token

    def _request(self, method: str, url: str, params: dict = None) -> dict:
        for attempt in range(6):
            token = self._get_token()
            headers = {"Authorization": f"Bearer {token}"}
            resp = requests.request(method, url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "2"))
                time.sleep(min(60, retry_after))
                continue
            if 200 <= resp.status_code < 300:
                return resp.json()
            # Retry on transient errors
            if resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            # Otherwise raise
            resp.raise_for_status()
        raise RuntimeError(f"Failed Spotify request after retries: {url}")

    def search_artist(self, name: str, limit: int = 5) -> List[dict]:
        q = f'artist:"{name}"'
        data = self._request("GET", "https://api.spotify.com/v1/search", params={"q": q, "type": "artist", "limit": limit})
        return data.get("artists", {}).get("items", [])

    def get_best_artist_match(self, name: str) -> Optional[dict]:
        candidates = self.search_artist(name, limit=10)
        if not candidates:
            return None
        target_norm = normalize_name(name)
        # 1) exact normalized name match
        for c in candidates:
            if normalize_name(c.get("name", "")) == target_norm:
                return c
        # 2) startswith
        for c in candidates:
            if normalize_name(c.get("name", "")).startswith(target_norm):
                return c
        # 3) fallback to highest popularity
        candidates.sort(key=lambda x: x.get("popularity", 0), reverse=True)
        return candidates[0]


def collect_unique_main_artists(data_dir: Path) -> List[str]:
    unique = set()
    for year in YEARS:
        p = data_dir / f"{year}song_lyrics_and_genres.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        artist_col, title_col, genre_col = detect_columns(df)
        if genre_col is None:
            genre_col = "genres"
            if genre_col not in df.columns:
                df[genre_col] = pd.NA
        is_missing = df[genre_col].isna() | (df[genre_col].astype(str).str.strip() == "") | (df[genre_col].astype(str).str.lower() == "nan")
        df_missing = df[is_missing].copy()
        if artist_col is None:
            # if no artist column, skip
            continue
        df_missing["__main_artist__"] = df_missing[artist_col].apply(extract_main_artist)
        for a in df_missing["__main_artist__"].dropna().unique():
            a = str(a).strip()
            if a:
                unique.add(a)
    return sorted(unique)


def load_cache(cache_path: Path) -> Dict[str, List[str]]:
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict[str, List[str]], cache_path: Path) -> None:
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def fetch_genres_for_artists(artists: List[str], client: SpotifyClient, cache_path: Path) -> Dict[str, List[str]]:
    cache = load_cache(cache_path)
    out = dict(cache)
    for name in artists:
        if name in out:
            continue
        best = client.get_best_artist_match(name)
        genres = best.get("genres", []) if best else []
        out[name] = genres
        # save incremental cache
        save_cache(out, cache_path)
        print(f"[cache] {name} -> {genres}")
    return out


def fill_csvs_with_genres(data_dir: Path, artist_genres: Dict[str, List[str]]) -> pd.DataFrame:
    records = []
    for year in YEARS:
        p = data_dir / f"{year}song_lyrics_and_genres.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        artist_col, title_col, genre_col = detect_columns(df)
        if genre_col is None:
            genre_col = "genres"
            if genre_col not in df.columns:
                df[genre_col] = pd.NA

        # mask for missing
        is_missing = (
            df[genre_col].isna()
            | (df[genre_col].astype(str).str.strip() == "")
            | (df[genre_col].astype(str).str.lower() == "nan")
        )
        before_missing = int(is_missing.sum())
        filled = 0

        if artist_col is not None:
            main_artists = df[artist_col].apply(extract_main_artist)

            # build fill series, aligned by index
            fill_series = main_artists.map(
                lambda a: "; ".join(artist_genres.get(a, []))
                if isinstance(a, str) and artist_genres.get(a)
                else pd.NA
            )

            # assign only where missing, alignment avoids length mismatch
            df[genre_col] = df[genre_col].where(~is_missing, fill_series)

            after_missing = int(
                (df[genre_col].isna() | (df[genre_col].astype(str).str.strip() == "")).sum()
            )
            filled = before_missing - after_missing
        else:
            after_missing = before_missing

        out_path = p.with_name(p.stem + "_genres_filled.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        records.append({
            "year": year,
            "input_file": str(p),
            "output_file": str(out_path),
            "missing_before": before_missing,
            "filled_from_spotify": filled,
            "missing_after": after_missing,
            "artist_col": artist_col or "",
            "genre_col": genre_col,
        })
        print(f"[write] {out_path} (filled {filled})")

    return pd.DataFrame.from_records(records)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=".", help="Directory containing *song_lyrics_and_genres.csv files")
    ap.add_argument("--client-id", type=str, default=os.environ.get("SPOTIFY_CLIENT_ID"))
    ap.add_argument("--client-secret", type=str, default=os.environ.get("SPOTIFY_CLIENT_SECRET"))
    args = ap.parse_args()

    if not args.client_id or not args.client_secret:
        raise SystemExit("Missing Spotify credentials. Provide --client-id/--client-secret or set SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET.")

    data_dir = Path(args.data_dir)
    cache_path = data_dir / CACHE_FILE

    # 1) Collect unique main artists
    artists = collect_unique_main_artists(data_dir)
    print(f"[artists] {len(artists)} main artists to fetch")

    # 2) Fetch genres w/ caching
    client = SpotifyClient(args.client_id, args.client_secret)
    artist_genres = fetch_genres_for_artists(artists, client, cache_path)

    # 3) Fill CSVs and write report
    report_df = fill_csvs_with_genres(data_dir, artist_genres)
    report_path = data_dir / REPORT_FILE
    report_df.to_csv(report_path, index=False)
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
