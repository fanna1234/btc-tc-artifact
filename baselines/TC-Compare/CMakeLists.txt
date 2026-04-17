cmake_minimum_required(VERSION 3.22)

# TC-Compare: A collection of triangle counting baselines
# Reference: TC-Compare benchmark suite

# Build preprocessing tools for format conversion
add_subdirectory(preprocessing)

# Note: TC-Compare approach binaries (tricore, fox, hu, etc.) are built
# using their original build scripts for baseline-faithfulness.
# See baselines/TC-Compare/approach/*/Makefile for individual builds.
