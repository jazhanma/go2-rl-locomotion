#!/usr/bin/env python3
"""
GitHub Sync Script with Duplicate File Cleanup

This script:
1. Scans the project directory for duplicate files (same name + extension)
2. Keeps only the most recently modified file, deletes older duplicates
3. Stages, commits, and pushes all changes to GitHub
4. Provides clear logging throughout the process

Usage: python3 sync_to_github.py
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict


def find_and_clean_duplicates(root_dir: str = ".") -> Dict[str, List[str]]:
    """
    Find and clean duplicate files in the project directory.
    
    Args:
        root_dir: Root directory to scan (default: current directory)
        
    Returns:
        Dictionary mapping file names to lists of duplicate paths
    """
    print("🔍 Scanning for duplicate files...")
    
    # Dictionary to store files by name+extension
    file_groups = defaultdict(list)
    
    # Scan directory recursively
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories and common build/cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'build', 'dist']]
        
        for file in files:
            if not file.startswith('.'):  # Skip hidden files
                file_path = os.path.join(root, file)
                file_groups[file].append(file_path)
    
    # Find duplicates (files with same name appearing in multiple locations)
    duplicates = {name: paths for name, paths in file_groups.items() if len(paths) > 1}
    
    if not duplicates:
        print("✅ No duplicate files found!")
        return {}
    
    print(f"📋 Found {len(duplicates)} files with duplicates:")
    
    cleaned_files = {}
    
    for file_name, file_paths in duplicates.items():
        print(f"\n📁 {file_name}:")
        
        # Sort by modification time (newest first)
        file_paths_with_time = []
        for path in file_paths:
            try:
                mtime = os.path.getmtime(path)
                file_paths_with_time.append((path, mtime))
            except OSError:
                print(f"  ⚠️  Could not access: {path}")
                continue
        
        file_paths_with_time.sort(key=lambda x: x[1], reverse=True)
        
        # Keep the newest file, delete the rest
        newest_file = file_paths_with_time[0][0]
        older_files = [path for path, _ in file_paths_with_time[1:]]
        
        print(f"  ✅ Keeping: {newest_file}")
        print(f"  🗑️  Deleting {len(older_files)} older duplicates:")
        
        for old_file in older_files:
            try:
                os.remove(old_file)
                print(f"    - {old_file}")
            except OSError as e:
                print(f"    ⚠️  Failed to delete {old_file}: {e}")
        
        cleaned_files[file_name] = {
            'kept': newest_file,
            'deleted': older_files
        }
    
    return cleaned_files


def git_sync() -> bool:
    """
    Stage, commit, and push changes to GitHub.
    
    Returns:
        True if successful, False otherwise
    """
    print("\n🔄 Starting Git synchronization...")
    
    try:
        # Check if git is initialized
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Git repository not initialized. Skipping Git operations.")
            return False
        
        # Check if there are any changes
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not result.stdout.strip():
            print("✅ No changes to commit. Repository is up to date.")
            return True
        
        # Stage all changes
        print("📦 Staging all changes...")
        result = subprocess.run(['git', 'add', '-A'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to stage changes: {result.stderr}")
            return False
        
        # Create commit message with current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Sync: {timestamp}"
        
        print(f"💾 Committing changes: '{commit_message}'")
        result = subprocess.run(['git', 'commit', '-m', commit_message], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to commit: {result.stderr}")
            return False
        
        print("✅ Changes committed successfully!")
        
        # Push to main branch
        print("🚀 Pushing to GitHub...")
        result = subprocess.run(['git', 'push', 'origin', 'main'], capture_output=True, text=True)
        if result.returncode != 0:
            if "fetch first" in result.stderr or "Updates were rejected" in result.stderr:
                print("⚠️  Remote has changes. Attempting to pull and merge...")
                pull_result = subprocess.run(['git', 'pull', 'origin', 'main'], capture_output=True, text=True)
                if pull_result.returncode == 0:
                    print("✅ Successfully pulled remote changes. Retrying push...")
                    push_result = subprocess.run(['git', 'push', 'origin', 'main'], capture_output=True, text=True)
                    if push_result.returncode == 0:
                        print("✅ Successfully pushed to GitHub after pull!")
                        return True
                    else:
                        print(f"❌ Failed to push after pull: {push_result.stderr}")
                        return False
                else:
                    print(f"❌ Failed to pull remote changes: {pull_result.stderr}")
                    return False
            else:
                print(f"❌ Failed to push: {result.stderr}")
                return False
        
        print("✅ Successfully pushed to GitHub!")
        return True
        
    except FileNotFoundError:
        print("❌ Git command not found. Please ensure Git is installed.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during Git operations: {e}")
        return False


def print_git_status() -> None:
    """Print current Git status for verification."""
    try:
        print("\n📊 Current Git Status:")
        result = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout.strip():
                print(result.stdout)
            else:
                print("  Working directory clean")
        else:
            print("  Could not retrieve Git status")
    except Exception as e:
        print(f"  Error retrieving Git status: {e}")


def main():
    """Main function to orchestrate duplicate cleanup and Git sync."""
    print("🎯 GitHub Sync Script with Duplicate Cleanup")
    print("=" * 50)
    
    # Step 1: Find and clean duplicates
    cleaned_files = find_and_clean_duplicates()
    
    if cleaned_files:
        print(f"\n🧹 Cleanup Summary:")
        print(f"  - Files with duplicates: {len(cleaned_files)}")
        total_deleted = sum(len(info['deleted']) for info in cleaned_files.values())
        print(f"  - Duplicate files deleted: {total_deleted}")
        print(f"  - Files kept: {len(cleaned_files)}")
    
    # Step 2: Git synchronization
    git_success = git_sync()
    
    # Step 3: Print final status
    print_git_status()
    
    # Final summary
    print("\n" + "=" * 50)
    if git_success:
        print("🎉 GitHub sync completed successfully!")
    else:
        print("⚠️  GitHub sync had issues, but duplicate cleanup completed.")
    
    if cleaned_files:
        print(f"🧹 Cleaned up {len(cleaned_files)} files with duplicates")
    
    print("✅ Script execution completed!")


if __name__ == "__main__":
    main()
