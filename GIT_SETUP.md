# Git Setup Guide

## 1. Initialize Git Repository
```bash
cd ~/fake-job-posting-detection
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## 2. Add All Files
```bash
git add .
```

## 3. Create Initial Commit
```bash
git commit -m "Initial project structure and documentation"
```

## 4. Create GitHub Repository
- Go to https://github.com/new
- Repository name: `fake-job-posting-detection`
- Description: "ML system to detect fraudulent job postings with interpretability analysis"
- Choose Public or Private
- Do NOT initialize with README (we already have one)
- Click "Create repository"

## 5. Add Remote and Push
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/fake-job-posting-detection.git
git branch -M main
git push -u origin main
```

## 6. Verify on GitHub
Visit: https://github.com/YOUR_USERNAME/fake-job-posting-detection
You should see all your files and folders!

## 7. Add Team Members
On GitHub:
1. Go to Settings → Collaborators
2. Click "Add people"
3. Search for each teammate's GitHub username
4. They'll receive an invitation to join

## Useful Git Commands Going Forward

```bash
# See status of changes
git status

# Create and switch to new branch
git checkout -b feature/your-feature-name

# Stage changes
git add .

# Commit with message
git commit -m "Describe your changes"

# Push to GitHub
git push origin feature/your-feature-name

# Pull latest changes from main
git pull origin main

# View commit history
git log --oneline
```
