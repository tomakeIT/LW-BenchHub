#!/bin/bash

# CI script to update workspace to a specific commit and run a command
# Usage: ./run_ci.sh <commit_id> <command>

set -e  # Exit on any error

# Check if both arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <commit_id> <command>"
    echo "Example: $0 abc123 'python test.py'"
    exit 1
fi

COMMIT_ID="$1"
shift  # Remove first argument, leaving the command and its arguments
COMMAND="$@"

echo "=== CI Script Started ==="
echo "Commit ID: $COMMIT_ID"
echo "Command: $COMMAND"
echo "Current directory: $(pwd)"

# Navigate to the lwlab workspace
cd /workspace/lwlab

echo "=== Updating to commit $COMMIT_ID ==="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Error: /workspace/lwlab is not a git repository"
    exit 1
fi

# Fetch latest changes
echo "Fetching latest changes..."
git fetch origin

# Check if the commit exists
if ! git rev-parse --verify "$COMMIT_ID" >/dev/null 2>&1; then
    echo "Error: Commit $COMMIT_ID not found"
    exit 1
fi

# Reset to the specified commit (hard reset to ensure clean state)
echo "Resetting to commit $COMMIT_ID..."
git reset --hard "$COMMIT_ID"

# Clean any untracked files (optional, but good for CI)
echo "Cleaning untracked files..."
git clean -fd

echo "=== Running command: $COMMAND ==="

# Run the provided command
eval "$COMMAND"

EXIT_CODE=$?

echo "=== Command completed with exit code: $EXIT_CODE ==="

exit $EXIT_CODE
