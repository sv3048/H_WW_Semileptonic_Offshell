#!/bin/bash
DATASET=$1
OUTPUT_DIR=$2

# Ensure Python finds modules in the current directory
export PYTHONPATH=.:$PYTHONPATH

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Call Analysis.py with the dataset name (no chunking)
python3 Analysis.py $DATASET

