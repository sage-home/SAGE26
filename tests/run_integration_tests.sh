#!/bin/bash
#
# Integration Test Runner
#
# This script runs the SAGE model with test parameters and validates outputs
# against expected behaviors.
#

set -e  # Exit on error

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  SAGE26 MODEL INTEGRATION TESTS${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Start timing
START_TIME=$(date +%s)

# Check if SAGE executable exists
if [ ! -f "$PROJECT_ROOT/sage" ]; then
    echo -e "${YELLOW}⚠ SAGE executable not found. Compile first with:${NC}"
    echo "   cd $PROJECT_ROOT && make"
    echo ""
fi

# Create test output directory
TEST_OUTPUT_DIR="$SCRIPT_DIR/test_output"
mkdir -p "$TEST_OUTPUT_DIR"

# Change to tests directory
cd "$SCRIPT_DIR"

# Helper function to run a test if it exists
# Args: test_number, test_name, test_binary, description, expected_test_count
run_test_if_exists() {
    local test_num=$1
    local test_name=$2
    local test_binary=$3
    local description=$4
    local expected_count=$5
    
    # Extract binary base name for log file (e.g., test_conservation from ./test_build/test_conservation)
    local binary_basename=$(basename "$test_binary")
    local log_file="$TEST_OUTPUT_DIR/${binary_basename}.log"
    
    if [ ! -f "$test_binary" ]; then
        echo -e "${YELLOW}▸ Test $test_num: $test_name ($expected_count tests)${NC}"
        echo -e "  ${YELLOW}⊗ SKIP - Test not built (missing dependencies)${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}▸ Test $test_num: $test_name ($expected_count tests)${NC}"
    echo -e "  ${NC}$description${NC}"
    
    if "$test_binary" > "$log_file" 2>&1; then
        local pass_info=$(grep -o "Passed:.*" "$log_file" | head -1)
        echo -e "${GREEN}  ✓ PASS - $pass_info${NC}"
        return 0
    else
        echo -e "${RED}  ✗ FAIL - Some ${test_name} tests failed${NC}"
        echo "  See: $log_file"
        return 0
    fi
}

echo -e "${YELLOW}▸ Test 1: Building all test suites${NC}"
echo -e "  ${NC}Compiling test executables...${NC}"
BUILD_START=$(date +%s)
make clean > /dev/null 2>&1
BUILD_OUTPUT=$(make all 2>&1)
BUILD_EXIT=$?
BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

# Count how many tests were actually built and failed
BUILT_COUNT=$(echo "$BUILD_OUTPUT" | grep "✓ Built" | wc -l | tr -d ' ')
FAILED_BUILD=$(echo "$BUILD_OUTPUT" | grep "✗ Failed to build" | wc -l | tr -d ' ')

if [ $BUILD_EXIT -eq 0 ]; then
    echo -e "${GREEN}  ✓ PASS - $BUILT_COUNT test suite(s) compiled successfully (${BUILD_TIME}s)${NC}"
    
    # Check if any tests were skipped or failed to build
    SKIPPED=$(echo "$BUILD_OUTPUT" | grep "⊗" | wc -l | tr -d ' ')
    if [ "$SKIPPED" -gt 0 ] || [ "$FAILED_BUILD" -gt 0 ]; then
        if [ "$SKIPPED" -gt 0 ]; then
            echo -e "${YELLOW}  ⚠ Note: $SKIPPED test(s) skipped due to missing dependencies${NC}"
        fi
        if [ "$FAILED_BUILD" -gt 0 ]; then
            echo -e "${YELLOW}  ⚠ Note: $FAILED_BUILD test(s) failed to build (compilation errors)${NC}"
        fi
    fi
else
    echo -e "${RED}  ✗ FAIL - Build failed${NC}"
    echo "$BUILD_OUTPUT"
    exit 1
fi

# If no tests were built at all, exit
if [ "$BUILT_COUNT" -eq 0 ]; then
    echo -e "${RED}  ✗ FAIL - No tests could be built${NC}"
    echo -e "  Run 'make check_dependencies' for details"
    exit 1
fi

run_test_if_exists 2 "Conservation Laws" "./test_build/test_conservation" \
    "Testing mass, metal, and energy conservation..." 37

run_test_if_exists 3 "Regime Determination & CGM Physics" "./test_build/test_regime_cgm" \
    "Testing Voit criterion, precipitation, and regime routing..." 21

run_test_if_exists 4 "Bulge Size Physics" "./test_build/test_bulge_sizes" \
    "Testing Shen scaling relations and instability bulges..." 13

run_test_if_exists 5 "Physics Validation" "./test_build/test_physics_validation" \
    "Testing physical bounds, quenching, and parameter ranges..." 31

run_test_if_exists 6 "Galaxy Mergers" "./test_build/test_mergers" \
    "Testing major/minor classification and merger timescales..." 13

run_test_if_exists 7 "Disk Instability" "./test_build/test_disk_instability" \
    "Testing stability criterion and bulge formation channel..." 9

run_test_if_exists 8 "Gas Infall" "./test_build/test_infall" \
    "Testing baryon fraction and regime-based gas routing..." 12

run_test_if_exists 9 "Numerical Stability" "./test_build/test_numerical_stability" \
    "Testing NaN/Inf protection and roundoff error handling..." 24

run_test_if_exists 10 "Metal Enrichment" "./test_build/test_metal_enrichment" \
    "Testing stellar yields, SN feedback, and metal tracking..." 22

run_test_if_exists 11 "Satellite Gas Stripping" "./test_build/test_stripping" \
    "Testing hot/cold gas stripping in satellites..." 14

run_test_if_exists 12 "Multi-Satellite Systems" "./test_build/test_multi_satellite" \
    "Testing orbital dynamics and tidal interactions..." 9

run_test_if_exists 13 "Star Formation Recipes" "./test_build/test_star_formation_recipes" \
    "Testing SF laws, quenching, and feedback efficiency..." 27

run_test_if_exists 14 "Reincorporation" "./test_build/test_reincorporation" \
    "Testing ejected gas return rates and timescales..." 21

run_test_if_exists 15 "Cooling & Heating" "./test_build/test_cooling_heating" \
    "Testing thermal balance, tcool/tff, and precipitation..." 25

run_test_if_exists 16 "Halo Assembly & Mergers" "./test_build/test_halo_mergers" \
    "Testing mass ratios, dynamical friction, and BH growth..." 24

run_test_if_exists 17 "AGN Feedback" "./test_build/test_agn_feedback" \
    "Testing radio/quasar modes, Eddington limits, and heating..." 18

run_test_if_exists 18 "Mimic L-Halo Reader" "./test_build/test_mimic_lhalo_reader" \
    "Testing the 88-byte record conversion and malformed-file rejection..." 97

# Count total passes, fails, and skipped tests
PASSED=0
FAILED=0
SUITES_RUN=0
SUITES_SKIPPED=0

# Check which test suites exist and sum up results from each log file
for test_exe in ./test_build/test_*; do
    # Skip .dSYM directories and non-executable files
    if [ -f "$test_exe" ] && [ -x "$test_exe" ] && [[ ! "$test_exe" =~ \.dSYM ]]; then
        SUITES_RUN=$((SUITES_RUN + 1))
        test_name=$(basename "$test_exe")
        log="$TEST_OUTPUT_DIR/${test_name}.log"
        if [ -f "$log" ]; then
            # Extract just the count number from "  Passed:       XX (YY%)" format
            P=$(grep "  Passed:" "$log" | sed 's/.*Passed:[^0-9]*\([0-9]*\).*/\1/')
            F=$(grep "  Failed:" "$log" | sed 's/.*Failed:[^0-9]*\([0-9]*\).*/\1/')
            if [ -n "$P" ] && [ "$P" != "Passed:" ]; then
                PASSED=$((PASSED + P))
            fi
            if [ -n "$F" ] && [ "$F" != "Failed:" ]; then
                FAILED=$((FAILED + F))
            fi
        fi
    fi
done

# Count expected test suites (count .c files)
TOTAL_SUITES=$(ls test_*.c 2>/dev/null | wc -l | tr -d ' ')
SUITES_SKIPPED=$((TOTAL_SUITES - SUITES_RUN))
TOTAL_TESTS=$((PASSED + FAILED))

# Calculate percentages using bc or python (avoid division by zero)
if [ "$TOTAL_TESTS" -gt 0 ]; then
    if command -v bc &> /dev/null; then
        PASS_PCT=$(echo "scale=1; ($PASSED * 100) / $TOTAL_TESTS" | bc)
        FAIL_PCT=$(echo "scale=1; ($FAILED * 100) / $TOTAL_TESTS" | bc)
    else
        PASS_PCT=$(python3 -c "print(f'{($PASSED/$TOTAL_TESTS)*100:.1f}')")
        FAIL_PCT=$(python3 -c "print(f'{($FAILED/$TOTAL_TESTS)*100:.1f}')")
    fi
else
    PASS_PCT="0.0"
    FAIL_PCT="0.0"
fi

# Calculate total execution time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  TEST SUMMARY${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "  Test Suites:  $SUITES_RUN of $TOTAL_SUITES run"
if [ "$SUITES_SKIPPED" -gt 0 ]; then
    echo -e "  ${YELLOW}Skipped:      $SUITES_SKIPPED suite(s) (missing dependencies)${NC}"
fi
echo -e "  Total tests:  $TOTAL_TESTS"
if [ "$TOTAL_TESTS" -gt 0 ]; then
    echo -e "  Passed:       $PASSED (${PASS_PCT}%)"
    echo -e "  Failed:       $FAILED (${FAIL_PCT}%)"
fi
echo -e "  Execution:    ${TOTAL_TIME}s"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

if [ "$TOTAL_TESTS" -eq 0 ]; then
    echo -e "${RED}  ✗ NO TESTS COULD BE RUN${NC}"
    echo ""
    echo -e "All test suites were skipped due to missing dependencies."
    echo -e "Run 'make check_dependencies' for details."
    echo ""
    exit 1
elif [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}  ✓ ALL INTEGRATION TESTS PASSED${NC}"
    echo ""
    if [ "$TOTAL_TESTS" -gt 0 ]; then
        echo -e "  Performance:  $TOTAL_TESTS tests in ${TOTAL_TIME}s ($(python3 -c "print(f'{$TOTAL_TESTS/$TOTAL_TIME:.1f}')") tests/sec)"
    fi
    echo -e "  Logs saved:   ${TEST_OUTPUT_DIR}"
    if [ "$SUITES_SKIPPED" -eq 0 ]; then
        echo -e "  Coverage:     Complete model physics validated"
    else
        echo -e "  ${YELLOW}Note:         $SUITES_SKIPPED suite(s) skipped${NC}"
    fi
    echo ""
    exit 0
else
    echo -e "${RED}  ✗ SOME TESTS FAILED${NC}"
    echo ""
    echo -e "Review logs in: ${TEST_OUTPUT_DIR}"
    exit 1
fi
