# GitHub Sync Script Usage Guide

## Overview
The `sync_to_github.py` script automatically cleans up duplicate files and syncs your project with GitHub.

## Features
- **Duplicate Cleanup**: Finds files with identical names and extensions, keeps the most recent version
- **Git Integration**: Stages, commits, and pushes changes automatically
- **Smart Conflict Resolution**: Handles merge conflicts by pulling remote changes first
- **Comprehensive Logging**: Clear output showing what's being cleaned and synced

## Usage

### Basic Usage
```bash
python3 sync_to_github.py
```

### What It Does
1. **Scans** the project directory recursively for duplicate files
2. **Cleans** duplicates by keeping only the most recently modified file
3. **Stages** all changes with `git add -A`
4. **Commits** with timestamp: "Sync: YYYY-MM-DD HH:MM:SS"
5. **Pushes** to the main branch on GitHub

### Duplicate Detection
Files are considered duplicates if they have:
- Same filename (including extension)
- Different locations in the project tree
- The script keeps the file with the most recent modification time

### Excluded Directories
The script automatically skips:
- Hidden directories (starting with `.`)
- `__pycache__` directories
- `node_modules`
- `build` and `dist` directories

## Output Example
```
🎯 GitHub Sync Script with Duplicate Cleanup
==================================================
🔍 Scanning for duplicate files...
📋 Found 2 files with duplicates:

📁 demo_recorder.py:
  ✅ Keeping: ./demo_recorder.py
  🗑️  Deleting 1 older duplicates:
    - ./examples/demo_recorder.py

📁 README.md:
  ✅ Keeping: ./README.md
  🗑️  Deleting 1 older duplicates:
    - ./examples/README.md

🧹 Cleanup Summary:
  - Files with duplicates: 2
  - Duplicate files deleted: 2
  - Files kept: 2

🔄 Starting Git synchronization...
📦 Staging all changes...
💾 Committing changes: 'Sync: 2025-09-16 16:56:26'
✅ Changes committed successfully!
🚀 Pushing to GitHub...
✅ Successfully pushed to GitHub!

📊 Current Git Status:
  Working directory clean

==================================================
🎉 GitHub sync completed successfully!
🧹 Cleaned up 2 files with duplicates
✅ Script execution completed!
```

## Error Handling
- **No Git Repository**: Warns and skips Git operations
- **No Changes**: Skips commit/push if working directory is clean
- **Merge Conflicts**: Automatically pulls remote changes and retries push
- **File Access Issues**: Reports files that couldn't be accessed or deleted

## Safety Features
- Only deletes files that are exact duplicates (same name + extension)
- Always keeps the most recent version
- Provides detailed logging of all operations
- Handles Git conflicts gracefully

## Requirements
- Python 3.6+
- Git installed and configured
- GitHub repository with remote origin set

## Tips
- Run this script regularly to keep your repository clean
- The script is safe to run multiple times
- Check the output to see what files were cleaned up
- Use `git status` to verify changes before running if needed
