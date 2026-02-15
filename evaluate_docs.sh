#!/bin/bash

# Script to evaluate documents for ALL category types: cv, motivation, presentations
# Usage:
#   ./evaluate_all_docs.sh                          # uses defaults
#   ./evaluate_all_docs.sh 20 "meta-llama/llama-3-70b-instruct"   # sets n=20 and model

# Configuration
API_KEY="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"  # ⚠️ REPLACE THIS!

# Parse optional arguments
NUM_CATEGORIES=${1:-100}
MODEL_NAME=${2:-"mistralai/mistral-nemo"}


# Check dependencies
if [[ ! -f "evaluate_documents_batched.py" ]]; then
    echo "ERROR: evaluate_documents_batched.py not found in current directory!"
    exit 1
fi

# Create output directory
mkdir -p output

# Define all category types
declare -A CATEGORY_MAP=(
    ["cv"]="cat_mining_results/cv_categories.json:data/raw/cvs_an:output/cv_scores_$(echo "$MODEL_NAME" | tr '/' '_').csv"
    ["motivation"]="cat_mining_results/motivation_categories.json:data/raw/letters_an:output/motivation_scores_$(echo "$MODEL_NAME" | tr '/' '_').csv"
    ["presentations"]="cat_mining_results/presentations_categories.json:data/raw/presentations_an:output/presentation_scores_$(echo "$MODEL_NAME" | tr '/' '_').csv"
)

echo "🚀 Starting batch evaluation for all document types..."
echo "================================"
echo "Number of categories: $NUM_CATEGORIES"
echo "Model: $MODEL_NAME"
echo "================================"

for TYPE in "cv" "motivation" "presentations"; do
    IFS=':' read -r CATEGORIES_FILE FOLDER OUTPUT <<< "${CATEGORY_MAP[$TYPE]}"
    
    echo
    echo "📄 Processing: $TYPE"
    echo "  Categories: $CATEGORIES_FILE"
    echo "  Folder: $FOLDER"
    echo "  Output: $OUTPUT"
    
    # Check if input folder exists
    if [[ ! -d "$FOLDER" ]]; then
        echo "  ⚠️ Warning: Folder '$FOLDER' not found. Skipping $TYPE."
        continue
    fi
    
    # Check if categories file exists
    if [[ ! -f "$CATEGORIES_FILE" ]]; then
        echo "  ⚠️ Warning: Categories file '$CATEGORIES_FILE' not found. Skipping $TYPE."
        continue
    fi
    
    # Run evaluation
    python3 evaluate_documents_batched.py \
        --categories "$CATEGORIES_FILE" \
        --folder "$FOLDER" \
        --output "$OUTPUT" \
        --n "$NUM_CATEGORIES" \
        --api_key "$API_KEY" \
        --model "$MODEL_NAME" \
        --batch_size 8
    
    if [[ $? -eq 0 ]]; then
        echo "  ✅ Completed: $TYPE → $OUTPUT"
    else
        echo "  ❌ Failed: $TYPE"
    fi
done

echo
echo "🎉 All evaluations completed!"