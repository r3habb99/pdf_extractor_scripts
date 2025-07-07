#!/usr/bin/env python3
"""
Visual Text Extraction Comparison Tool

This script provides side-by-side visual comparison of raw text extraction
between Python and Node.js systems for better understanding of differences.
"""

import json
import re
from pathlib import Path

def load_python_raw_text(file_path: str) -> str:
    """Load raw text from Python extraction JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('raw_text', '')

def load_nodejs_raw_text(file_path: str) -> str:
    """Load raw text from Node.js extraction JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pages = data.get('document', {}).get('pages', [])
    if pages:
        return pages[0].get('text', '')
    return ''

def show_side_by_side_comparison(python_text: str, nodejs_text: str, title: str, lines_to_show: int = 50):
    """Show side-by-side comparison of text extractions."""
    print(f"\n{'=' * 120}")
    print(f"{title.center(120)}")
    print(f"{'=' * 120}")
    
    python_lines = python_text.split('\n')[:lines_to_show]
    nodejs_lines = nodejs_text.split('\n')[:lines_to_show]
    
    max_lines = max(len(python_lines), len(nodejs_lines))
    
    print(f"{'PYTHON EXTRACTION (pdfplumber)'.center(60)} | {'NODE.JS EXTRACTION (pdf-text-extract)'.center(60)}")
    print(f"{'-' * 60} | {'-' * 60}")
    
    for i in range(max_lines):
        python_line = python_lines[i] if i < len(python_lines) else ""
        nodejs_line = nodejs_lines[i] if i < len(nodejs_lines) else ""
        
        # Truncate lines if too long
        python_display = python_line[:58] + ".." if len(python_line) > 60 else python_line
        nodejs_display = nodejs_line[:58] + ".." if len(nodejs_line) > 60 else nodejs_line
        
        print(f"{python_display:<60} | {nodejs_display:<60}")

def analyze_line_item_structure(python_text: str, nodejs_text: str, title: str):
    """Analyze line item structure differences."""
    print(f"\n{'=' * 120}")
    print(f"LINE ITEM STRUCTURE ANALYSIS - {title}".center(120))
    print(f"{'=' * 120}")
    
    # Find line items in both texts
    line_item_pattern = re.compile(r'^\s*(\d{3})\s+.*?(\d+\.\d+)\s+(\d+\.\d+)\s*$', re.MULTILINE)
    
    python_items = line_item_pattern.findall(python_text)
    nodejs_items = line_item_pattern.findall(nodejs_text)
    
    print(f"Python line items found: {len(python_items)}")
    print(f"Node.js line items found: {len(nodejs_items)}")
    
    print(f"\nFirst 5 line items comparison:")
    print(f"{'PYTHON':<40} | {'NODE.JS':<40}")
    print(f"{'-' * 40} | {'-' * 40}")
    
    max_items = min(5, max(len(python_items), len(nodejs_items)))
    for i in range(max_items):
        python_item = f"{python_items[i]}" if i < len(python_items) else "N/A"
        nodejs_item = f"{nodejs_items[i]}" if i < len(nodejs_items) else "N/A"
        
        print(f"{str(python_item)[:38]:<40} | {str(nodejs_item)[:38]:<40}")

def analyze_spacing_patterns(python_text: str, nodejs_text: str, title: str):
    """Analyze spacing pattern differences."""
    print(f"\n{'=' * 120}")
    print(f"SPACING PATTERN ANALYSIS - {title}".center(120))
    print(f"{'=' * 120}")
    
    # Find lines with significant spacing
    spacing_pattern = re.compile(r'.*\s{5,}.*')
    
    python_spaced_lines = [line for line in python_text.split('\n') if spacing_pattern.match(line)]
    nodejs_spaced_lines = [line for line in nodejs_text.split('\n') if spacing_pattern.match(line)]
    
    print(f"Python lines with 5+ spaces: {len(python_spaced_lines)}")
    print(f"Node.js lines with 5+ spaces: {len(nodejs_spaced_lines)}")
    
    print(f"\nSample spaced lines (first 3):")
    print(f"{'PYTHON':<60} | {'NODE.JS':<60}")
    print(f"{'-' * 60} | {'-' * 60}")
    
    max_samples = min(3, max(len(python_spaced_lines), len(nodejs_spaced_lines)))
    for i in range(max_samples):
        python_line = python_spaced_lines[i][:58] + ".." if i < len(python_spaced_lines) and len(python_spaced_lines[i]) > 60 else (python_spaced_lines[i] if i < len(python_spaced_lines) else "N/A")
        nodejs_line = nodejs_spaced_lines[i][:58] + ".." if i < len(nodejs_spaced_lines) and len(nodejs_spaced_lines[i]) > 60 else (nodejs_spaced_lines[i] if i < len(nodejs_spaced_lines) else "N/A")
        
        print(f"{python_line:<60} | {nodejs_line:<60}")

def analyze_numeric_precision(python_text: str, nodejs_text: str, title: str):
    """Analyze numeric precision differences."""
    print(f"\n{'=' * 120}")
    print(f"NUMERIC PRECISION ANALYSIS - {title}".center(120))
    print(f"{'=' * 120}")
    
    numeric_pattern = re.compile(r'\d+\.\d+')
    
    python_numbers = numeric_pattern.findall(python_text)
    nodejs_numbers = numeric_pattern.findall(nodejs_text)
    
    # Find unique numbers in each system
    python_set = set(python_numbers)
    nodejs_set = set(nodejs_numbers)
    
    only_in_python = python_set - nodejs_set
    only_in_nodejs = nodejs_set - python_set
    common_numbers = python_set & nodejs_set
    
    print(f"Total numbers in Python: {len(python_numbers)}")
    print(f"Total numbers in Node.js: {len(nodejs_numbers)}")
    print(f"Unique numbers in Python: {len(python_set)}")
    print(f"Unique numbers in Node.js: {len(nodejs_set)}")
    print(f"Common numbers: {len(common_numbers)}")
    print(f"Only in Python: {len(only_in_python)}")
    print(f"Only in Node.js: {len(only_in_nodejs)}")
    
    if only_in_python:
        print(f"\nNumbers only in Python (first 10): {list(only_in_python)[:10]}")
    if only_in_nodejs:
        print(f"Numbers only in Node.js (first 10): {list(only_in_nodejs)[:10]}")

def main():
    """Main visual comparison function."""
    print("=" * 120)
    print("VISUAL PDF TEXT EXTRACTION COMPARISON".center(120))
    print("=" * 120)
    
    # File paths
    ceco_python = "output/ceco/raw_text/CECO F3GU7A-I-01793287_raw_text_ceco.json"
    ceco_nodejs = "backend/output/370555-001_CECO F3GU7A-I-01793287_ceco.json"
    
    steelcraft_python = "output/steelcraft/raw_text/Allegion - Steelcraft_raw_text_steelcraft.json"
    steelcraft_nodejs = "backend/output/364805-02_Allegion - Steelcraft_steelcraft.json"
    
    # CECO Comparison
    if Path(ceco_python).exists() and Path(ceco_nodejs).exists():
        print("\n🏢 CECO INVOICE VISUAL COMPARISON")
        
        ceco_python_text = load_python_raw_text(ceco_python)
        ceco_nodejs_text = load_nodejs_raw_text(ceco_nodejs)
        
        # Show first 30 lines side by side
        show_side_by_side_comparison(ceco_python_text, ceco_nodejs_text, "CECO INVOICE - FIRST 30 LINES", 30)
        
        # Analyze line item structure
        analyze_line_item_structure(ceco_python_text, ceco_nodejs_text, "CECO")
        
        # Analyze spacing patterns
        analyze_spacing_patterns(ceco_python_text, ceco_nodejs_text, "CECO")
        
        # Analyze numeric precision
        analyze_numeric_precision(ceco_python_text, ceco_nodejs_text, "CECO")
    
    # Steelcraft Comparison
    if Path(steelcraft_python).exists() and Path(steelcraft_nodejs).exists():
        print("\n🏗️ STEELCRAFT INVOICE VISUAL COMPARISON")
        
        steelcraft_python_text = load_python_raw_text(steelcraft_python)
        steelcraft_nodejs_text = load_nodejs_raw_text(steelcraft_nodejs)
        
        # Show first 30 lines side by side
        show_side_by_side_comparison(steelcraft_python_text, steelcraft_nodejs_text, "STEELCRAFT INVOICE - FIRST 30 LINES", 30)
        
        # Analyze line item structure
        analyze_line_item_structure(steelcraft_python_text, steelcraft_nodejs_text, "STEELCRAFT")
        
        # Analyze spacing patterns
        analyze_spacing_patterns(steelcraft_python_text, steelcraft_nodejs_text, "STEELCRAFT")
        
        # Analyze numeric precision
        analyze_numeric_precision(steelcraft_python_text, steelcraft_nodejs_text, "STEELCRAFT")
    
    print(f"\n{'=' * 120}")
    print("VISUAL COMPARISON COMPLETE".center(120))
    print(f"{'=' * 120}")

if __name__ == "__main__":
    main()
