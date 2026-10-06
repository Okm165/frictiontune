# Common developer tasks. Install just: https://github.com/casey/just

set shell := ["bash", "-uc"]
root := justfile_directory()

# Default: show the recipe list.
default:
    @just --list

# Install the light data pipeline (CPU box).
sync:
    uv sync --all-packages

# Install the GPU training side too.
sync-train:
    uv sync --package frictiontune-train --extra train

# Developer tooling (ruff, mypy, pytest, pre-commit) comes from the workspace
# `dev` dependency group, so a normal sync installs it.
sync-dev:
    uv sync --all-packages

# Lint every package and the test suite.
lint:
    uv run ruff check packages tests

# Check formatting without writing.
fmt-check:
    uv run ruff format --check packages tests

# Apply formatting.
fmt:
    uv run ruff format packages tests

# Strict type check both packages and the test suite.
typecheck:
    uv run mypy packages/frictiontune/src/frictiontune packages/frictiontune-train/src/frictiontune_train tests

# Run the fast test suite.
test:
    uv run pytest

# Everything CI should run.
check: lint fmt-check typecheck test

# Validate the default configuration.
validate:
    uv run frictiontune validate --config-dir configs

# List every stage and its ROI contract.
stages:
    uv run frictiontune stages

# Dry-run the core pipeline against a hardware preset.
dry-run preset="3x24g":
    uv run frictiontune run --hardware {{preset}} --dry-run

# Run the core pipeline.
run preset="3x24g":
    uv run frictiontune run --hardware {{preset}}

# Explore the artifacts of the last run.
inspect:
    uv run frictiontune inspect

# Render the run report.
report:
    uv run frictiontune report

# Render the friction-drop dashboard.
dashboard:
    uv run frictiontune dashboard