#!/bin/bash

# Default settings
VERBOSE=false
TEST_TYPE="all"
PYTEST_ARGS=""

# Set up colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Help message
show_help() {
    echo -e "${YELLOW}Usage: ./run_tests.sh [options]${NC}"
    echo "Options:"
    echo "  --help           Show this help message"
    echo "  --verbose, -v    Run tests in verbose mode"
    echo "  --all            Run all tests (default)"
    echo "  --unit           Run only unit tests"
    echo "  --integration    Run only integration tests"
    echo "  --extra PATTERN  Run tests matching the given pattern"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --all)
            TEST_TYPE="all"
            shift
            ;;
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --extra)
            TEST_TYPE="extra"
            if [[ -z "$2" || "$2" == --* ]]; then
                echo -e "${RED}Error: --extra requires a pattern argument${NC}"
                exit 1
            fi
            PATTERN="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Set up verbose flag if needed
if $VERBOSE; then
    PYTEST_ARGS="$PYTEST_ARGS -v"
fi

# Run the appropriate tests
case $TEST_TYPE in
    all)
        echo -e "${YELLOW}Running all tests...${NC}"
        python -m pytest $PYTEST_ARGS
        ;;
    unit)
        echo -e "${YELLOW}Running unit tests only...${NC}"
        python -m pytest -k "not integration" $PYTEST_ARGS
        ;;
    integration)
        echo -e "${YELLOW}Running integration tests only...${NC}"
        python -m pytest tests/test_integration.py $PYTEST_ARGS
        ;;
    extra)
        echo -e "${YELLOW}Running tests matching pattern: $PATTERN${NC}"
        python -m pytest -k "$PATTERN" $PYTEST_ARGS
        ;;
esac

# Check the test result
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Tests failed with exit code $exit_code${NC}"
    exit $exit_code
fi
