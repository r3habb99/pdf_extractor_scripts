"""
Batch PDF Processing Module

This module provides functionality to process multiple PDFs in batch
with consistent output quality and performance optimization.
"""

import os
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict

from fallback_processor import FallbackProcessor, ProcessingResult
from output_consolidator import OutputConsolidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Container for batch processing results."""
    total_files: int
    successful: int
    failed: int
    processing_time: float
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    summary: Dict[str, Any]


class BatchProcessor:
    """
    Processes multiple PDFs in batch with optimization and error handling.
    """
    
    def __init__(
        self,
        output_dir: str = "./output",
        max_workers: int = 4,
        save_individual_files: bool = True,
        consolidate_outputs: bool = True,
        **processor_kwargs
    ):
        """
        Initialize the batch processor.

        Args:
            output_dir: Directory for output files
            max_workers: Maximum number of concurrent workers
            save_individual_files: Whether to save individual JSON files
            consolidate_outputs: Whether to consolidate multiple outputs per PDF
            **processor_kwargs: Additional arguments for FallbackProcessor
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.max_workers = max_workers
        self.save_individual_files = save_individual_files
        self.consolidate_outputs = consolidate_outputs

        # Initialize the fallback processor
        self.processor = FallbackProcessor(output_dir=str(self.output_dir), **processor_kwargs)

        # Initialize the output consolidator
        if self.consolidate_outputs:
            self.consolidator = OutputConsolidator()

        logger.info(f"BatchProcessor initialized with {max_workers} workers, consolidation: {consolidate_outputs}")
    
    def process_directory(
        self,
        input_dir: str,
        file_pattern: str = "*.pdf",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """
        Process all PDF files in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            file_pattern: File pattern to match (default: "*.pdf")
            progress_callback: Optional callback function for progress updates
            
        Returns:
            BatchResult containing processing results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all PDF files
        pdf_files = list(input_path.glob(file_pattern))
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir} matching pattern {file_pattern}")
            return BatchResult(
                total_files=0,
                successful=0,
                failed=0,
                processing_time=0.0,
                results=[],
                errors=[],
                summary={"message": "No files found"}
            )
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        return self.process_files([str(p) for p in pdf_files], progress_callback)
    
    def process_files(
        self,
        file_paths: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """
        Process a list of PDF files.
        
        Args:
            file_paths: List of PDF file paths
            progress_callback: Optional callback function for progress updates
            
        Returns:
            BatchResult containing processing results
        """
        start_time = time.time()
        
        # Convert to Path objects
        pdf_files = [Path(p) for p in file_paths]
        total_files = len(pdf_files)
        
        results = []
        errors = []
        successful = 0
        failed = 0
        
        logger.info(f"Starting batch processing of {total_files} files")
        
        # Process files with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, pdf_file): pdf_file
                for pdf_file in pdf_files
            }
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_file)):
                pdf_file = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    if result.success:
                        successful += 1
                        results.append({
                            'file_path': str(pdf_file),
                            'status': 'success',
                            'data': result.data,
                            'method_used': result.method_used.value if result.method_used else None,
                            'confidence_score': result.confidence_score,
                            'processing_time': result.processing_time,
                            'warnings': result.warnings
                        })
                        
                        # Save individual file if requested
                        if self.save_individual_files:
                            self._save_individual_result(pdf_file, result)

                            # Consolidate outputs if enabled
                            if self.consolidate_outputs:
                                self._consolidate_pdf_outputs(pdf_file, result)
                            
                    else:
                        failed += 1
                        error_info = {
                            'file_path': str(pdf_file),
                            'status': 'failed',
                            'errors': result.errors,
                            'processing_time': result.processing_time
                        }
                        errors.append(error_info)
                        logger.error(f"Failed to process {pdf_file.name}: {result.errors}")
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(i + 1, total_files)
                        
                except Exception as e:
                    failed += 1
                    error_info = {
                        'file_path': str(pdf_file),
                        'status': 'failed',
                        'errors': [f"Unexpected error: {str(e)}"],
                        'processing_time': None
                    }
                    errors.append(error_info)
                    logger.error(f"Unexpected error processing {pdf_file.name}: {e}")
        
        processing_time = time.time() - start_time
        
        # Create batch result
        batch_result = BatchResult(
            total_files=total_files,
            successful=successful,
            failed=failed,
            processing_time=processing_time,
            results=results,
            errors=errors,
            summary=self._create_batch_summary(total_files, successful, failed, processing_time)
        )
        
        logger.info(f"Batch processing completed: {successful}/{total_files} successful "
                   f"in {processing_time:.2f}s")
        
        return batch_result
    
    def _process_single_file(self, pdf_file: Path) -> ProcessingResult:
        """
        Process a single PDF file.
        
        Args:
            pdf_file: Path to the PDF file
            
        Returns:
            ProcessingResult
        """
        try:
            logger.info(f"Processing {pdf_file.name}")
            return self.processor.process_pdf(pdf_file)
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            return ProcessingResult(
                success=False,
                errors=[f"Processing error: {str(e)}"]
            )
    
    def _save_individual_result(self, pdf_file: Path, result: ProcessingResult) -> None:
        """
        Save individual processing result to JSON file in vendor-specific folder.

        Args:
            pdf_file: Original PDF file path
            result: Processing result
        """
        try:
            # Determine vendor-specific output folder
            output_folder = self._get_vendor_output_folder(pdf_file, result)
            output_folder.mkdir(parents=True, exist_ok=True)

            output_file = output_folder / f"{pdf_file.stem}_extracted.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result.data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved result to vendor-specific folder: {output_file}")
        except Exception as e:
            logger.error(f"Error saving individual result for {pdf_file.name}: {e}")

    def _get_vendor_output_folder(self, pdf_file: Path, result: ProcessingResult) -> Path:
        """
        Determine the appropriate output folder based on vendor detection.

        Args:
            pdf_file: Original PDF file path
            result: Processing result

        Returns:
            Path to vendor-specific output folder
        """
        # Try to get vendor info from result metadata
        metadata = result.data.get('metadata', {}) if result.data else {}
        vendor_folder = metadata.get('vendor_folder')

        if vendor_folder:
            return self.output_dir.parent / vendor_folder

        # Fallback: detect vendor from filename
        from vendor_detector import detect_vendor_type
        vendor_result = detect_vendor_type(pdf_file)
        vendor_folder = vendor_result.get('output_folder', 'output/unknown')

        return self.output_dir.parent / vendor_folder

    def _consolidate_pdf_outputs(self, pdf_file: Path, result: ProcessingResult) -> None:
        """
        Consolidate multiple output files for a single PDF.

        Args:
            pdf_file: Original PDF file path
            result: Processing result
        """
        try:
            # Get the vendor output folder
            vendor_folder = self._get_vendor_output_folder(pdf_file, result)

            # Wait a moment to ensure all files are written
            import time
            time.sleep(0.1)

            # Consolidate outputs for this specific PDF
            pdf_name = self.consolidator._extract_pdf_name(f"{pdf_file.stem}_extracted.json")
            file_groups = self.consolidator._group_files_by_pdf(vendor_folder)

            if pdf_name in file_groups and len(file_groups[pdf_name]) > 1:
                consolidation_result = self.consolidator._consolidate_pdf_outputs(
                    pdf_name, file_groups[pdf_name], vendor_folder
                )

                if consolidation_result["status"] == "consolidated":
                    logger.info(f"Consolidated {len(file_groups[pdf_name])} files for {pdf_name}: "
                              f"{consolidation_result['action']} (accuracy: {consolidation_result['final_accuracy']:.1f}%)")
                else:
                    logger.debug(f"No consolidation needed for {pdf_name}")

        except Exception as e:
            logger.error(f"Error consolidating outputs for {pdf_file.name}: {e}")
    

    def _create_batch_summary(
        self,
        total_files: int,
        successful: int,
        failed: int,
        processing_time: float
    ) -> Dict[str, Any]:
        """
        Create batch processing summary.
        
        Args:
            total_files: Total number of files processed
            successful: Number of successful extractions
            failed: Number of failed extractions
            processing_time: Total processing time
            
        Returns:
            Summary dictionary
        """
        success_rate = (successful / total_files * 100) if total_files > 0 else 0
        avg_time_per_file = processing_time / total_files if total_files > 0 else 0
        
        return {
            "batch_statistics": {
                "total_files": total_files,
                "successful_extractions": successful,
                "failed_extractions": failed,
                "success_rate_percent": round(success_rate, 2),
                "total_processing_time_seconds": round(processing_time, 2),
                "average_time_per_file_seconds": round(avg_time_per_file, 2)
            },
            "processing_info": {
                "max_workers": self.max_workers,
                "output_directory": str(self.output_dir),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    
    def get_processing_statistics(self, batch_result: BatchResult) -> Dict[str, Any]:
        """
        Get detailed processing statistics from batch result.
        
        Args:
            batch_result: Batch processing result
            
        Returns:
            Detailed statistics dictionary
        """
        if not batch_result.results:
            return {"message": "No successful results to analyze"}
        
        # Analyze extraction methods used
        methods_used = {}
        confidence_scores = []
        processing_times = []
        
        for result in batch_result.results:
            method = result.get('method_used', 'unknown')
            methods_used[method] = methods_used.get(method, 0) + 1
            
            if result.get('confidence_score'):
                confidence_scores.append(result['confidence_score'])
            
            if result.get('processing_time'):
                processing_times.append(result['processing_time'])
        
        # Calculate statistics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "extraction_methods": methods_used,
            "quality_metrics": {
                "average_confidence_score": round(avg_confidence, 2),
                "min_confidence_score": min(confidence_scores) if confidence_scores else None,
                "max_confidence_score": max(confidence_scores) if confidence_scores else None
            },
            "performance_metrics": {
                "average_processing_time_seconds": round(avg_processing_time, 2),
                "min_processing_time_seconds": min(processing_times) if processing_times else None,
                "max_processing_time_seconds": max(processing_times) if processing_times else None
            }
        }


def process_pdf_batch(input_dir: str, **kwargs) -> BatchResult:
    """
    Convenience function to process a batch of PDFs.
    
    Args:
        input_dir: Directory containing PDF files
        **kwargs: Additional arguments for BatchProcessor
        
    Returns:
        BatchResult
    """
    processor = BatchProcessor(**kwargs)
    return processor.process_directory(input_dir)
