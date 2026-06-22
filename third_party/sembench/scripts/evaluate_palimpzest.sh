#!/bin/bash

# evaluate_palimpzest.sh - Evaluate Palimpzest with different configurations
# Usage: ./evaluate_palimpzest.sh

# Note: set -e is disabled to handle os._exit(0) from run.py
# which can interfere with bash script execution

# Configuration
USE_CASE="movie"  # Focus on movie use case for evaluation
SYSTEM="palimpzest"
QUERIES=(1 2 3 4 5 6 7 8 9 10)
BASE_DIR="files"

# Configuration files to test
CONFIG_FILES=(
    "gemini-2.0-flash-maxquality.json"
    "gemini-2.0-flash-mincost.json"
    "gemini-2.5-flash-maxquality.json"
    "gemini-2.5-flash-mincost.json"
    "gpt-4o-mini-maxquality.json"
    "gpt-4o-mini-mincost.json"
    "gpt-5-mini-maxquality.json"
    "gpt-5-mini-mincost.json"
    "mixed-models-maxquality.json"
    "mixed-models-mincost.json"
)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Palimpzest Configuration Evaluation ===${NC}"
echo "Evaluating ${#CONFIG_FILES[@]} different configurations..."
echo "Use case: ${USE_CASE}"
echo "System: ${SYSTEM}"
echo ""

# Create evaluation directory
EVAL_DIR="${BASE_DIR}/${USE_CASE}/metrics/palimpzest/evaluate"
mkdir -p "${EVAL_DIR}"
echo -e "${YELLOW}Created evaluation directory: ${EVAL_DIR}${NC}"
echo ""

# Function to extract config name from filename
get_config_name() {
    local config_file=$1
    # Remove .json extension to get config name
    echo "${config_file%.json}"
}

# Function to run evaluation for a single configuration
run_evaluation() {
    local config_file=$1
    local config_name=$(get_config_name "$config_file")
    
    echo -e "${GREEN}=== Evaluating Configuration: ${config_name} ===${NC}"
    echo "Config file: ${config_file}"
    echo "Environment variable will be: PALIMPZEST_CONFIG_FILE=${config_file}"
    echo "Running:"
    echo "  PALIMPZEST_CONFIG_FILE=${config_file} python3 src/run.py --systems ${SYSTEM} --use-cases ${USE_CASE} --queries ${QUERIES[*]}"
    
    # Set environment variable for config file and run
    export PALIMPZEST_CONFIG_FILE="${config_file}"
    python3 src/run.py \
        --systems "${SYSTEM}" \
        --use-cases "${USE_CASE}" \
        --queries "${QUERIES[@]}"
    
    local exit_code=$?
    unset PALIMPZEST_CONFIG_FILE
    
    if [[ ${exit_code} -eq 0 ]]; then
        echo -e "${GREEN}✓ Completed evaluation for ${config_name}${NC}"
        
        # Copy results to evaluation directory
        local source_file="${BASE_DIR}/${USE_CASE}/metrics/${SYSTEM}.json"
        local target_file="${EVAL_DIR}/${config_name}.json"
        
        if [[ -f "${source_file}" ]]; then
            cp "${source_file}" "${target_file}"
            echo -e "${GREEN}✓ Results saved to ${target_file}${NC}"
        else
            echo -e "${RED}⚠ Warning: Results file not found at ${source_file}${NC}"
        fi
    else
        echo -e "${RED}✗ Failed evaluation for ${config_name}${NC}"
        return 1
    fi
    echo ""
}

# Main execution loop
echo -e "${BLUE}Starting evaluations...${NC}"
echo ""

successful_configs=0
failed_configs=0

for config_file in "${CONFIG_FILES[@]}"; do
    if run_evaluation "${config_file}"; then
        ((successful_configs++))
    else
        ((failed_configs++))
        echo -e "${RED}Evaluation failed for ${config_file}. Continuing with next configuration...${NC}"
    fi
    
    echo -e "${BLUE}────────────────────────────────────${NC}"
done

echo -e "${GREEN}=== Evaluation Complete! ===${NC}"
echo "Successful evaluations: ${successful_configs}"
echo "Failed evaluations: ${failed_configs}"
echo ""
echo "Results stored in:"
echo "  ${EVAL_DIR}/"
echo ""
echo "Available result files:"
ls -la "${EVAL_DIR}/" | grep -E "\.json$" || echo "  No result files found"