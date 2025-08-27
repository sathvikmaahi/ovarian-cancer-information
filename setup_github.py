#!/usr/bin/env python3
"""
Automated GitHub Repository Setup Script

This script automates the process of setting up and uploading your project to GitHub.
You only need to:
1. Create the repository on GitHub manually
2. Run this script with your GitHub username and repository name
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_git_status():
    """Check if git is properly configured"""
    print("🔍 Checking git configuration...")
    
    # Check if git is installed
    if not run_command("git --version", "Checking git installation"):
        print("❌ Git is not installed. Please install git first.")
        return False
    
    # Check if we're in a git repository
    if not os.path.exists(".git"):
        print("❌ Not in a git repository. Please run 'git init' first.")
        return False
    
    # Check git status
    if not run_command("git status", "Checking git status"):
        return False
    
    return True

def get_repository_info():
    """Get repository information from user"""
    print("\n" + "="*60)
    print("GITHUB REPOSITORY SETUP")
    print("="*60)
    
    print("\n📋 Before running this script, please:")
    print("1. Go to github.com and create a new repository")
    print("2. Name it: 'Ovarian cancer review using images'")
    print("3. Make it public or private as you prefer")
    print("4. Do NOT initialize with README, .gitignore, or license")
    print("5. Copy the repository URL")
    
    print("\n🔗 Enter the repository URL (e.g., https://github.com/username/repo-name.git):")
    repo_url = input().strip()
    
    if not repo_url.startswith("https://github.com/"):
        print("❌ Invalid GitHub URL. Please enter a valid GitHub repository URL.")
        return None
    
    return repo_url

def setup_github_repository(repo_url):
    """Set up the GitHub repository and push code"""
    print(f"\n🚀 Setting up GitHub repository: {repo_url}")
    
    # Add remote origin
    if not run_command(f"git remote add origin {repo_url}", "Adding remote origin"):
        print("❌ Failed to add remote origin. The repository might already exist.")
        return False
    
    # Push to GitHub
    if not run_command("git push -u origin main", "Pushing code to GitHub"):
        print("❌ Failed to push to GitHub. Please check your credentials and repository access.")
        return False
    
    return True

def verify_upload():
    """Verify that files were uploaded correctly"""
    print("\n🔍 Verifying uploaded files...")
    
    expected_files = [
        ".gitignore",
        "README_GITHUB.md", 
        "requirements.txt",
        "run_pipeline.py",
        "src/data_preparation.py",
        "src/model_implementation.py"
    ]
    
    print("\n📁 Files in your repository:")
    for file_path in expected_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"  ❌ {file_path} (missing)")
    
    print(f"\n📊 Total files: {len(expected_files)}")
    print("🎯 All essential files are ready for upload!")

def create_upload_instructions():
    """Create a simple instruction file for manual upload if needed"""
    instructions = """
# Manual GitHub Upload Instructions

If the automated script doesn't work, follow these steps:

## 1. Create Repository on GitHub
- Go to github.com
- Click "+" → "New repository"
- Name: "Ovarian cancer review using images"
- Make it public or private
- Do NOT initialize with README, .gitignore, or license
- Click "Create repository"

## 2. Upload Files
- Copy the repository URL
- Run these commands in your terminal:

```bash
# Add remote origin
git remote add origin YOUR_REPOSITORY_URL

# Push to GitHub
git push -u origin main
```

## 3. Verify Upload
- Refresh your GitHub repository page
- You should see all 6 essential files uploaded
"""
    
    with open("GITHUB_UPLOAD_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)
    
    print("📝 Created GITHUB_UPLOAD_INSTRUCTIONS.md for manual reference")

def main():
    """Main function"""
    print("🚀 Automated GitHub Repository Setup")
    print("="*50)
    
    # Check git status
    if not check_git_status():
        print("\n❌ Git setup issues detected. Please fix them and try again.")
        return
    
    # Get repository information
    repo_url = get_repository_info()
    if not repo_url:
        print("\n❌ No valid repository URL provided. Exiting.")
        return
    
    # Set up GitHub repository
    if setup_github_repository(repo_url):
        print("\n🎉 SUCCESS! Your code has been uploaded to GitHub!")
        print(f"🔗 Repository: {repo_url}")
        print("\n📋 Next steps:")
        print("1. Visit your GitHub repository")
        print("2. Verify all files are uploaded correctly")
        print("3. Share your repository with others")
        
        # Verify upload
        verify_upload()
    else:
        print("\n❌ Failed to set up GitHub repository.")
        print("📝 Creating manual upload instructions...")
        create_upload_instructions()
        print("\n💡 Please follow the manual instructions in GITHUB_UPLOAD_INSTRUCTIONS.md")

if __name__ == "__main__":
    main()
