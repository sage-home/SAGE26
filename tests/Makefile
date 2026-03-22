# Makefile for SAGE Model Unit Tests
#
# This Makefile compiles unit tests for the SAGE semi-analytic model
# with CGM physics extensions.
#
# Usage:
#   make              # Build all tests
#   make test         # Build and run all tests
#   make clean        # Clean build artifacts
#   make test_name    # Build specific test

# Compiler settings
CC = gcc
ROOT_DIR := $(shell cd .. && pwd)
CFLAGS = -std=gnu99 -Wall -Wextra -g -O2 -I../src -DROOT_DIR='"$(ROOT_DIR)"'
LDFLAGS = -lm

# Directories
SRC_DIR = ../src
TEST_DIR = .
OUTPUT_DIR = ./test_build

# Source files from main project (needed for tests)
PROJECT_SOURCES = \
	$(SRC_DIR)/core_utils.c \
	$(SRC_DIR)/model_misc.c \
	$(SRC_DIR)/model_starformation_and_feedback.c \
	$(SRC_DIR)/model_cooling_heating.c \
	$(SRC_DIR)/model_reincorporation.c \
	$(SRC_DIR)/model_infall.c \
	$(SRC_DIR)/model_disk_instability.c \
	$(SRC_DIR)/model_mergers.c \
	$(SRC_DIR)/core_cool_func.c

# Test files
TEST_SOURCES = \
	test_conservation.c \
	test_regime_cgm.c \
	test_bulge_sizes.c \
	test_physics_validation.c \
	test_mergers.c \
	test_disk_instability.c \
	test_infall.c \
	test_numerical_stability.c \
	test_metal_enrichment.c \
	test_stripping.c \
	test_multi_satellite.c \
	test_star_formation_recipes.c \
	test_reincorporation.c \
	test_cooling_heating.c \
	test_halo_mergers.c \
	test_agn_feedback.c \
	test_ffb_concentration.c

# Function to check if all dependencies exist for a test
# Usage: $(call check_test_deps,test_name)
define check_test_deps
$(shell \
	test_file="$(TEST_DIR)/$(1).c"; \
	if [ ! -f "$$test_file" ]; then \
		echo ""; \
		exit 0; \
	fi; \
	missing=""; \
	for src in $(PROJECT_SOURCES); do \
		if [ ! -f "$$src" ]; then \
			missing="$$missing $$src"; \
		fi; \
	done; \
	for header in $(SRC_DIR)/core_allvars.h $(SRC_DIR)/core_utils.h; do \
		if [ ! -f "$$header" ]; then \
			missing="$$missing $$header"; \
		fi; \
	done; \
	if [ -n "$$missing" ]; then \
		echo ""; \
	else \
		echo "$(OUTPUT_DIR)/$(1)"; \
	fi \
)
endef

# Build list of tests that can actually be built
BUILDABLE_TESTS = $(foreach test,$(TEST_SOURCES:%.c=%),$(call check_test_deps,$(test)))

# Test executables (only buildable ones)
TEST_BINS = $(BUILDABLE_TESTS)

# Default target - continue building even if some tests fail
all: check_dependencies
	@failed=0; built=0; \
	for test_src in $(TEST_SOURCES); do \
		test_name=$${test_src%.c}; \
		test_file="$(TEST_DIR)/$$test_src"; \
		if [ ! -f "$$test_file" ]; then \
			continue; \
		fi; \
		missing=""; \
		for src in $(PROJECT_SOURCES); do \
			if [ ! -f "$$src" ]; then \
				missing="$$missing $$src"; \
			fi; \
		done; \
		if [ -z "$$missing" ]; then \
			if $(MAKE) $(OUTPUT_DIR)/$$test_name 2>/dev/null; then \
				built=$$((built + 1)); \
			else \
				echo "✗ Failed to build $$test_name (compilation errors)"; \
				failed=$$((failed + 1)); \
			fi; \
		fi; \
	done; \
	echo ""; \
	if [ $$built -eq 0 ]; then \
		echo "✗ No tests were successfully built"; \
		exit 1; \
	else \
		echo "✓ Built $$built test(s)"; \
		if [ $$failed -gt 0 ]; then \
			echo "⚠ $$failed test(s) failed to build"; \
		fi; \
	fi

# Check and report on dependencies
check_dependencies:
	@echo "Checking test dependencies..."
	@total=0; buildable=0; \
	for test_src in $(TEST_SOURCES); do \
		total=$$((total + 1)); \
		test_name=$${test_src%.c}; \
		test_file="$(TEST_DIR)/$$test_src"; \
		if [ ! -f "$$test_file" ]; then \
			echo "  ⊗ $$test_name - test file missing"; \
			continue; \
		fi; \
		missing=""; \
		for src in $(PROJECT_SOURCES); do \
			if [ ! -f "$$src" ]; then \
				missing="$$missing $${src##*/}"; \
			fi; \
		done; \
		if [ -n "$$missing" ]; then \
			echo "  ⊗ $$test_name - missing:$$missing"; \
		else \
			echo "  ✓ $$test_name"; \
			buildable=$$((buildable + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "Tests buildable: $$buildable/$$total"; \
	echo ""

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Pattern rule for building tests
$(OUTPUT_DIR)/test_%: test_%.c $(PROJECT_SOURCES) | $(OUTPUT_DIR)
	@echo "Building $@..."
	@missing=""; \
	for src in $(PROJECT_SOURCES); do \
		if [ ! -f "$$src" ]; then \
			missing="$$missing $$src"; \
		fi; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "✗ Cannot build $@ - missing source files:$$missing"; \
		exit 1; \
	fi
	$(CC) $(CFLAGS) $< $(PROJECT_SOURCES) $(LDFLAGS) -o $@
	@echo "✓ Built $@"

# Run all tests
test: $(TEST_BINS)
	@if [ -z "$(TEST_BINS)" ]; then \
		echo ""; \
		echo "════════════════════════════════════════════════════════════"; \
		echo "  ⚠ NO TESTS TO RUN"; \
		echo "════════════════════════════════════════════════════════════"; \
		echo "  All tests are missing dependencies."; \
		echo "  Run 'make check_dependencies' for details."; \
		echo "════════════════════════════════════════════════════════════"; \
		echo ""; \
		exit 1; \
	fi; \
	echo ""; \
	echo "════════════════════════════════════════════════════════════"; \
	echo "  RUNNING ALL TESTS ($(words $(TEST_BINS)) suites)"; \
	echo "════════════════════════════════════════════════════════════"; \
	failed=0; run=0; \
	for test in $(TEST_BINS); do \
		if [ -f "$$test" ]; then \
			echo ""; \
			run=$$((run + 1)); \
			$$test || failed=$$((failed + 1)); \
		fi; \
	done; \
	echo ""; \
	if [ $$failed -eq 0 ]; then \
		echo "════════════════════════════════════════════════════════════"; \
		echo "  ✓ ALL $$run TEST SUITE(S) PASSED"; \
		echo "════════════════════════════════════════════════════════════"; \
		echo ""; \
	else \
		echo "════════════════════════════════════════════════════════════"; \
		echo "  ✗ $$failed of $$run TEST SUITE(S) FAILED"; \
		echo "════════════════════════════════════════════════════════════"; \
		echo ""; \
		exit 1; \
	fi

# Run individual test suites
test_conservation: $(OUTPUT_DIR)/test_conservation
	$(OUTPUT_DIR)/test_conservation

test_regime: $(OUTPUT_DIR)/test_regime_cgm
	$(OUTPUT_DIR)/test_regime_cgm

test_bulge: $(OUTPUT_DIR)/test_bulge_sizes
	$(OUTPUT_DIR)/test_bulge_sizes

test_physics: $(OUTPUT_DIR)/test_physics_validation
	$(OUTPUT_DIR)/test_physics_validation

test_ffb: $(OUTPUT_DIR)/test_ffb_concentration
	$(OUTPUT_DIR)/test_ffb_concentration

# Quick test (just conservation laws - fastest)
quick: $(OUTPUT_DIR)/test_conservation
	$(OUTPUT_DIR)/test_conservation

# Clean build artifacts
clean:
	rm -rf $(OUTPUT_DIR)
	@echo "✓ Cleaned build artifacts"

# Help target
help:
	@echo "SAGE26 Test Suite"
	@echo "================="
	@echo ""
	@echo "Comprehensive unit tests for SAGE26 semi-analytic model"
	@echo "with CGM physics extensions (278 tests across 16 suites)"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build all tests"
	@echo "  make test         - Build and run all tests"
	@echo "  make quick        - Run quick tests (conservation only)"
	@echo "  make test_conservation - Run conservation law tests"
	@echo "  make test_regime  - Run regime/CGM physics tests"
	@echo "  make test_bulge   - Run bulge size tests"
	@echo "  make test_physics - Run physics validation tests"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Documentation:"
	@echo "  README.md              - Test suite overview and usage"
	@echo "  TEST_SUITE_REVIEW.md   - Recent review findings (Dec 2025)"
	@echo "  run_integration_tests.sh - Full integration tests"
	@echo ""

.PHONY: all test quick clean help test_conservation test_regime test_bulge test_physics