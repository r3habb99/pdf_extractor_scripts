#!/usr/bin/env python3
"""
Output Consolidation System

This module consolidates multiple JSON extraction outputs for the same PDF into a single
best file based on accuracy scores and data completeness. It handles:

1. Comparing accuracy/confidence scores between files
2. Merging data when files have different extracted fields
3. Keeping only the highest quality output file
4. Removing duplicate/lower quality files

Usage:
    from output_consolidator import OutputConsolidator
    
    consolidator = OutputConsolidator()
    consolidator.consolidate_vendor_outputs("output/schlage")
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputConsolidator:
    """Consolidates multiple JSON extraction outputs into single best files."""
    
    def __init__(self):
        """Initialize the consolidator."""
        self.backup_dir = Path("output/backup")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def consolidate_vendor_outputs(self, vendor_folder: str) -> Dict[str, Any]:
        """
        Consolidate all outputs in a vendor folder.
        
        Args:
            vendor_folder: Path to vendor folder (e.g., "output/schlage")
            
        Returns:
            Summary of consolidation results
        """
        vendor_path = Path(vendor_folder)
        if not vendor_path.exists():
            logger.warning(f"Vendor folder does not exist: {vendor_folder}")
            return {"status": "error", "message": "Folder not found"}
        
        logger.info(f"Consolidating outputs in: {vendor_folder}")
        
        # Group files by PDF name
        file_groups = self._group_files_by_pdf(vendor_path)
        
        results = {
            "vendor_folder": vendor_folder,
            "total_pdfs": len(file_groups),
            "consolidated": 0,
            "skipped": 0,
            "errors": [],
            "details": []
        }
        
        for pdf_name, files in file_groups.items():
            try:
                if len(files) > 1:
                    # Multiple files found - consolidate them
                    consolidation_result = self._consolidate_pdf_outputs(pdf_name, files, vendor_path)
                    results["details"].append(consolidation_result)
                    
                    if consolidation_result["status"] == "consolidated":
                        results["consolidated"] += 1
                    else:
                        results["skipped"] += 1
                else:
                    # Single file - no consolidation needed
                    results["skipped"] += 1
                    results["details"].append({
                        "pdf_name": pdf_name,
                        "status": "single_file",
                        "message": "Only one output file found"
                    })
                    
            except Exception as e:
                error_msg = f"Error consolidating {pdf_name}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        logger.info(f"Consolidation complete: {results['consolidated']} consolidated, {results['skipped']} skipped")
        return results
    
    def _group_files_by_pdf(self, vendor_path: Path) -> Dict[str, List[Path]]:
        """
        Group JSON files by their source PDF name.
        
        Args:
            vendor_path: Path to vendor folder
            
        Returns:
            Dictionary mapping PDF names to lists of JSON files
        """
        file_groups = {}
        
        # Find all JSON files (excluding raw_text folder)
        json_files = [f for f in vendor_path.glob("*.json") if f.is_file()]
        
        for json_file in json_files:
            # Extract PDF name from filename
            pdf_name = self._extract_pdf_name(json_file.name)
            
            if pdf_name not in file_groups:
                file_groups[pdf_name] = []
            file_groups[pdf_name].append(json_file)
        
        return file_groups
    
    def _extract_pdf_name(self, filename: str) -> str:
        """
        Extract the original PDF name from a JSON filename.

        Args:
            filename: JSON filename

        Returns:
            Original PDF name
        """
        # Remove common suffixes to get base PDF name
        suffixes_to_remove = [
            "_schlage_extracted.json",
            "_steelcraft_extracted.json",
            "_ceco_extracted.json",
            "_extracted.json"  # This should be last to catch the generic case
        ]

        base_name = filename
        for suffix in suffixes_to_remove:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break

        return base_name

    def _consolidate_pdf_outputs(self, pdf_name: str, files: List[Path], vendor_path: Path) -> Dict[str, Any]:
        """
        Consolidate multiple output files for a single PDF.

        Args:
            pdf_name: Name of the source PDF
            files: List of JSON files for this PDF
            vendor_path: Path to vendor folder

        Returns:
            Consolidation result dictionary
        """
        logger.info(f"Consolidating {len(files)} files for PDF: {pdf_name}")

        # Load and analyze all files
        file_data = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                file_info = {
                    "path": file_path,
                    "data": data,
                    "accuracy_score": self._get_accuracy_score(data),
                    "confidence_score": self._get_confidence_score(data),
                    "data_completeness": self._calculate_data_completeness(data),
                    "file_type": self._determine_file_type(file_path.name)
                }
                file_data.append(file_info)

            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                continue

        if not file_data:
            return {
                "pdf_name": pdf_name,
                "status": "error",
                "message": "No valid files could be loaded"
            }

        # Determine the best file and merge strategy
        best_file, merge_needed = self._determine_best_file_and_merge_strategy(file_data)

        if merge_needed:
            # Merge data from multiple files
            merged_data = self._merge_file_data(file_data, best_file)
            final_data = merged_data
            action = "merged_and_consolidated"
        else:
            # Use the best file as-is
            final_data = best_file["data"]
            action = "selected_best"

        # Save the consolidated file
        consolidated_file = vendor_path / f"{pdf_name}_extracted.json"

        # Backup existing files before overwriting
        self._backup_files(files, pdf_name)

        # Save the final consolidated data
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        # Remove the other files
        files_removed = []
        for file_path in files:
            if file_path != consolidated_file:
                try:
                    file_path.unlink()
                    files_removed.append(file_path.name)
                except Exception as e:
                    logger.error(f"Error removing file {file_path}: {e}")

        return {
            "pdf_name": pdf_name,
            "status": "consolidated",
            "action": action,
            "final_file": consolidated_file.name,
            "final_accuracy": self._get_accuracy_score(final_data),
            "final_confidence": self._get_confidence_score(final_data),
            "files_removed": files_removed,
            "original_files": len(files)
        }

    def _get_accuracy_score(self, data: Dict[str, Any]) -> float:
        """Extract accuracy score from data."""
        metadata = data.get("metadata", {})
        return float(metadata.get("accuracy_score", 0.0))

    def _get_confidence_score(self, data: Dict[str, Any]) -> float:
        """Extract confidence score from data."""
        metadata = data.get("metadata", {})
        return float(metadata.get("confidence_score", 0.0))

    def _calculate_data_completeness(self, data: Dict[str, Any]) -> float:
        """
        Calculate data completeness score based on extracted fields.

        Args:
            data: Extracted data dictionary

        Returns:
            Completeness score (0-100)
        """
        # Count non-empty fields in key sections
        sections_to_check = [
            "invoice_header",
            "vendor_info",
            "customer_info",
            "line_items",
            "totals"
        ]

        total_fields = 0
        filled_fields = 0

        for section in sections_to_check:
            if section in data:
                section_data = data[section]
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        total_fields += 1
                        if value and str(value).strip():
                            filled_fields += 1
                elif isinstance(section_data, list) and section_data:
                    # For line_items, count as filled if list is not empty
                    total_fields += 1
                    filled_fields += 1

        if total_fields == 0:
            return 0.0

        return (filled_fields / total_fields) * 100

    def _determine_file_type(self, filename: str) -> str:
        """Determine the type of extraction file."""
        if "_schlage_extracted.json" in filename:
            return "vendor_specific"
        elif "_steelcraft_extracted.json" in filename:
            return "vendor_specific"
        elif "_ceco_extracted.json" in filename:
            return "vendor_specific"
        elif "_extracted.json" in filename:
            return "generic"
        else:
            return "unknown"

    def _determine_best_file_and_merge_strategy(self, file_data: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], bool]:
        """
        Determine the best file and whether merging is needed.

        Args:
            file_data: List of file information dictionaries

        Returns:
            Tuple of (best_file_info, merge_needed)
        """
        # Sort files by quality score (combination of accuracy, confidence, and completeness)
        def quality_score(file_info):
            accuracy = file_info["accuracy_score"]
            confidence = file_info["confidence_score"]
            completeness = file_info["data_completeness"]

            # Weighted score: accuracy is most important, then completeness, then confidence
            return (accuracy * 0.5) + (completeness * 0.3) + (confidence * 0.2)

        sorted_files = sorted(file_data, key=quality_score, reverse=True)
        best_file = sorted_files[0]

        # Check if merging would be beneficial
        merge_needed = False

        if len(sorted_files) > 1:
            # Check if other files have significantly different data that could be merged
            best_completeness = best_file["data_completeness"]

            for other_file in sorted_files[1:]:
                # If another file has reasonable quality and different data, consider merging
                other_quality = quality_score(other_file)
                best_quality = quality_score(best_file)

                # Merge if the other file has at least 70% of the best file's quality
                # and has some unique data (different completeness pattern)
                if (other_quality >= best_quality * 0.7 and
                    abs(other_file["data_completeness"] - best_completeness) > 10):
                    merge_needed = True
                    break

        logger.info(f"Best file: {best_file['path'].name} (quality: {quality_score(best_file):.1f})")
        logger.info(f"Merge needed: {merge_needed}")

        return best_file, merge_needed

    def _merge_file_data(self, file_data: List[Dict[str, Any]], best_file: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge data from multiple files, using the best file as base.

        Args:
            file_data: List of all file information
            best_file: The best file to use as base

        Returns:
            Merged data dictionary
        """
        logger.info("Merging data from multiple files...")

        # Start with the best file's data
        merged_data = best_file["data"].copy()

        # Merge data from other files
        for file_info in file_data:
            if file_info["path"] == best_file["path"]:
                continue  # Skip the base file

            other_data = file_info["data"]
            merged_data = self._deep_merge_data(merged_data, other_data)

        # Update metadata to reflect the merge
        if "metadata" not in merged_data:
            merged_data["metadata"] = {}

        merged_data["metadata"].update({
            "consolidation_timestamp": datetime.now().isoformat(),
            "consolidation_method": "accuracy_based_merge",
            "source_files": [f["path"].name for f in file_data],
            "final_accuracy_score": max(f["accuracy_score"] for f in file_data),
            "final_confidence_score": max(f["confidence_score"] for f in file_data)
        })

        return merged_data

    def _deep_merge_data(self, base_data: Dict[str, Any], other_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two data dictionaries, preferring non-empty values.

        Args:
            base_data: Base data dictionary
            other_data: Other data dictionary to merge in

        Returns:
            Merged data dictionary
        """
        merged = base_data.copy()

        for key, value in other_data.items():
            if key not in merged:
                # Key doesn't exist in base, add it
                merged[key] = value
            elif not merged[key] and value:
                # Base value is empty/None, use other value
                merged[key] = value
            elif isinstance(merged[key], dict) and isinstance(value, dict):
                # Both are dictionaries, merge recursively
                merged[key] = self._deep_merge_data(merged[key], value)
            elif isinstance(merged[key], list) and isinstance(value, list):
                # Both are lists, merge if other has more items
                if len(value) > len(merged[key]):
                    merged[key] = value
            # Otherwise, keep the base value (it's presumably better quality)

        return merged

    def _backup_files(self, files: List[Path], pdf_name: str) -> None:
        """
        Backup files before consolidation.

        Args:
            files: List of files to backup
            pdf_name: Name of the source PDF
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = self.backup_dir / f"{pdf_name}_{timestamp}"
        backup_folder.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            try:
                backup_path = backup_folder / file_path.name
                shutil.copy2(file_path, backup_path)
                logger.debug(f"Backed up {file_path.name} to {backup_path}")
            except Exception as e:
                logger.error(f"Error backing up {file_path}: {e}")


def main():
    """Main function for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate PDF extraction outputs")
    parser.add_argument("vendor_folder", nargs='?', help="Path to vendor folder (e.g., output/schlage)")
    parser.add_argument("--all", action="store_true", help="Consolidate all vendor folders")

    args = parser.parse_args()

    consolidator = OutputConsolidator()

    if args.all:
        # Consolidate all vendor folders
        vendor_folders = ["output/schlage", "output/steelcraft", "output/ceco"]
        for folder in vendor_folders:
            if Path(folder).exists():
                print(f"\n{'='*50}")
                print(f"Consolidating: {folder}")
                print('='*50)
                result = consolidator.consolidate_vendor_outputs(folder)
                print(f"Results: {result['consolidated']} consolidated, {result['skipped']} skipped")
                if result['errors']:
                    print(f"Errors: {len(result['errors'])}")
            else:
                print(f"Skipping {folder} (not found)")
    elif args.vendor_folder:
        # Consolidate specific folder
        result = consolidator.consolidate_vendor_outputs(args.vendor_folder)
        print(f"Consolidation complete:")
        print(f"  - PDFs processed: {result['total_pdfs']}")
        print(f"  - Files consolidated: {result['consolidated']}")
        print(f"  - Files skipped: {result['skipped']}")

        if result['errors']:
            print(f"  - Errors: {len(result['errors'])}")
            for error in result['errors']:
                print(f"    {error}")

        # Show details
        for detail in result['details']:
            if detail['status'] == 'consolidated':
                print(f"\n✅ {detail['pdf_name']}: {detail['action']}")
                print(f"   Final accuracy: {detail['final_accuracy']:.1f}%")
                print(f"   Files removed: {', '.join(detail['files_removed'])}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
