#!/bin/bash
# Download Open Food Facts CSV dump

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$PROJECT_ROOT/data/raw/openfoodfacts"
mkdir -p "$DEST"

OFF_URL="https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
OUTPUT="$DEST/products.csv.gz"

echo "Downloading Open Food Facts (~2GB compressed)..."
wget -c -O "$OUTPUT" "$OFF_URL"
echo "Decompressing..."
gunzip -k "$OUTPUT"
echo "Download complete: $DEST/products.csv"
