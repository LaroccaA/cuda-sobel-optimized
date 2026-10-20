#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT:-gpu-2D2D.py}"
EXPECT_GLOB="${EXPECT_GLOB:-*_sobel_baseline_16x16.png}"
PYTHON="${PYTHON:-python3}"

if [[ ! -d "$REPO_DIR/Dataset" ]]; then
  echo "ERRORE: manca $REPO_DIR/Dataset" >&2
  exit 1
fi
if [[ ! -f "$REPO_DIR/$SCRIPT" ]]; then
  echo "ERRORE: script non trovato: $REPO_DIR/$SCRIPT" >&2
  exit 1
fi

src="$(find "$REPO_DIR/Dataset" -maxdepth 1 -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
  -print -quit)"
if [[ -z "$src" ]]; then
  echo "ERRORE: nessuna immagine in $REPO_DIR/Dataset" >&2
  exit 1
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/sobel-smoke.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/Dataset"
cp "$src" "$tmp/Dataset/"

(
  cd "$tmp"
  "$PYTHON" "$REPO_DIR/$SCRIPT"
)

if [[ ! -d "$tmp/Output" ]]; then
  echo "ERRORE: Output/ non è stato creato" >&2
  exit 1
fi

out="$(find "$tmp/Output" -maxdepth 1 -type f \
  -name "$EXPECT_GLOB" -size +0c -print -quit)"
if [[ -z "$out" ]]; then
  echo "ERRORE: output non trovato ($EXPECT_GLOB) in $tmp/Output" >&2
  exit 1
fi

echo "OK: $out"
