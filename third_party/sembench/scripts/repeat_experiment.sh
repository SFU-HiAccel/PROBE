#!/bin/bash

# experiment.sh - Run benchmark experiments and collect results
# Usage: ./experiment.sh

set -e

# Configuration
USE_CASES=("animals")
SYSTEMS=("lotus")
QUERIES=(1 3 7 10)
NUM_ROUNDS=5
BASE_DIR="files"
MODEL_TAG="2.5flash"   # used in the target folder name

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MMBench Experiment Runner ===${NC}"
echo "Running benchmark for ${NUM_ROUNDS} rounds..."
echo "Use cases: ${USE_CASES[*]}"
echo "Systems: ${SYSTEMS[*]}"
echo ""

# Collect metrics for a single use case after all systems have run
collect_metrics_for_use_case() {
    local round_num=$1
    local use_case=$2

    echo -e "${YELLOW}Collecting metrics for round ${round_num} | use case: ${use_case}...${NC}"

    local metrics_dir="${BASE_DIR}/${use_case}/metrics"
    local target_dir="${metrics_dir}/across_system_${MODEL_TAG}_${round_num}"

    mkdir -p "${target_dir}"

    for system in "${SYSTEMS[@]}"; do
        local source_file="${metrics_dir}/${system}.json"
        if [[ -f "${source_file}" ]]; then
            cp "${source_file}" "${target_dir}/"
            echo "  ✓ Collected ${use_case}/${system}.json → ${target_dir}/"
        else
            echo -e "  ${RED}⚠ Warning: ${use_case}/${system}.json not found${NC}"
        fi
    done
    echo ""
}

# Run one system on one use case (single invocation)
run_single() {
    local round_num=$1
    local use_case=$2
    local system=$3

    echo -e "${GREEN}=== Round ${round_num}/${NUM_ROUNDS} | Use case: ${use_case} | System: ${system} ===${NC}"
    echo "Running:"
    echo "  python3 src/run.py --systems ${system} --use-cases ${use_case} --queries ${QUERIES[*]}"

    python3 src/run.py \
        --systems "${system}" \
        --use-cases "${use_case}" \
        --queries "${QUERIES[@]}"

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ Completed ${system} on ${use_case}${NC}"
    else
        echo -e "${RED}✗ Failed ${system} on ${use_case}${NC}"
        return 1
    fi
    echo ""
}

# Main execution: round → use case → system
for round in $(seq 1 ${NUM_ROUNDS}); do
    echo -e "${BLUE}Starting round ${round}...${NC}"

    for use_case in "${USE_CASES[@]}"; do
        echo -e "${BLUE}-- Use case: ${use_case} --${NC}"

        for system in "${SYSTEMS[@]}"; do
            if ! run_single "${round}" "${use_case}" "${system}"; then
                echo -e "${RED}Round ${round} aborted on ${use_case}/${system}.${NC}"
                exit 1
            fi
        done

        # After all systems for this use case, collect the metrics into the round folder
        collect_metrics_for_use_case "${round}" "${use_case}"
        echo -e "${GREEN}✓ Finished all systems for use case '${use_case}' in round ${round}${NC}"
        echo -e "${BLUE}────────────────────────────────────${NC}"
    done

    echo -e "${GREEN}Round ${round} completed successfully!${NC}"
    if [[ ${round} -lt ${NUM_ROUNDS} ]]; then
        echo ""
    fi
done

echo -e "${GREEN}=== Experiment Complete! ===${NC}"
echo "All ${NUM_ROUNDS} rounds completed successfully."
echo ""
echo "Results stored in (per use case):"
for use_case in "${USE_CASES[@]}"; do
    echo "  ${BASE_DIR}/${use_case}/metrics/across_system_${MODEL_TAG}_[1-${NUM_ROUNDS}]/"
done
