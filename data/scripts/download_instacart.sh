#!/bin/bash
# Download Instacart 2017 dataset from Kaggle
# Requires: kaggle CLI configured with ~/.kaggle/kaggle.json

set -e

DEST="/scratch/smehta90/Clickless AI/data/raw/instacart_2017"
mkdir -p "$DEST"

echo "Downloading Instacart Online Grocery Shopping 2017..."

if command -v kaggle &> /dev/null; then
    kaggle datasets download -d psparks/instacart-market-basket-analysis -p "$DEST" --unzip
    echo "Download complete: $DEST"
else
    echo "kaggle CLI not found. Manual download instructions:"
    echo "  1. Go to https://www.kaggle.com/c/instacart-market-basket-analysis/data"
    echo "  2. Download all CSV files"
    echo "  3. Place them in: $DEST"
    echo ""
    echo "Expected files:"
    echo "  orders.csv, order_products__prior.csv, order_products__train.csv"
    echo "  products.csv, aisles.csv, departments.csv"
fi
