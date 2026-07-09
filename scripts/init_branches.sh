#!/usr/bin/env bash
set -euo pipefail

git init -q
git add .
git commit -q -m "Initial commit"
git branch -M main
git branch develop
for b in feature/data-io feature/augmentations feature/models; do
  git branch "$b"
done
git branch
