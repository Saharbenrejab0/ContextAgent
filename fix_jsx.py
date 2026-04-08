"""
fix_jsx.py
Corrige les backticks échappés (\`) dans tous les fichiers JSX.
Lance depuis la racine du projet ContextAgent.
"""

import os
from pathlib import Path

jsx_dirs = [
    "frontend/src/pages",
    "frontend/src/components",
    "frontend/src",
]

fixed = 0

for d in jsx_dirs:
    for file in Path(d).glob("*.jsx"):
        content = file.read_text(encoding="utf-8")
        new_content = content.replace("\\`", "`")
        if new_content != content:
            file.write_text(new_content, encoding="utf-8")
            print(f"  fixed: {file}")
            fixed += 1

print(f"\nDone. {fixed} files fixed.")