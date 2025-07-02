#!/usr/bin/env python3
"""
Robust PDF Processing System

Main entry point for the robust PDF processing system that handles both
text-selectable and image-based PDFs with automatic detection and fallback mechanisms.

Usage:
    python robust_pdf_processor.py <pdf_path> [options]
    python robust_pdf_processor.py --batch <directory> [options]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from fallback_processor import FallbackProcessor, ProcessingResult
from batch_processor import BatchProcessor, BatchResult
from pdf_text_detector import detect_pdf_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RobustPDFProcessor:
    """
    Main unified PDF processor that orchestrates all components.
    Provides a clean API for PDF processing with automatic method selection.
    """
    
    def __init__(
        self,
        output_dir: str = "./output",
        min_confidence_threshold: float = 70.0,
        enable_hybrid_mode: bool = True,
        max_workers: int = 4,
        save_individual_files: bool = True,
        consolidate_outputs: bool = True,
        save_images: bool = False,
        **kwargs
    ):
        """
        Initialize the robust PDF processor.

        Args:
            output_dir: Directory for output files
            min_confidence_threshold: Minimum confidence score to accept results
            enable_hybrid_mode: Whether to enable hybrid processing
            max_workers: Maximum workers for batch processing
            save_individual_files: Whether to save individual JSON files
            consolidate_outputs: Whether to consolidate multiple outputs per PDF
            save_images: Whether to save intermediate PNG images during OCR processing
            **kwargs: Additional configuration options
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize processors
        self.fallback_processor = FallbackProcessor(
            output_dir=str(self.output_dir),
            min_confidence_threshold=min_confidence_threshold,
            enable_hybrid_mode=enable_hybrid_mode,
            save_images=save_images,
            **kwargs
        )

        self.batch_processor = BatchProcessor(
            output_dir=str(self.output_dir),
            max_workers=max_workers,
            save_individual_files=save_individual_files,
            consolidate_outputs=consolidate_outputs,
            min_confidence_threshold=min_confidence_threshold,
            enable_hybrid_mode=enable_hybrid_mode,
            save_images=save_images,
            **kwargs
        )

        logger.info("RobustPDFProcessor initialized")
    
    def process_single_pdf(
        self,
        pdf_path: str,
        output_file: Optional[str] = None,
        return_json: bool = False
    ) -> Any:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_file: Optional output file path
            return_json: Whether to return JSON string instead of dict
            
        Returns:
            Processing result dictionary or JSON string
        """
        logger.info(f"Processing single PDF: {pdf_path}")
        
        # Process the PDF
        result = self.fallback_processor.process_pdf(Path(pdf_path))
        
        if result.success:
            # Save to file if output_file specified
            if output_file:
                self._save_result_to_file(result.data, output_file)
            
            # Return result
            if return_json:
                return json.dumps(result.data, indent=2, ensure_ascii=False)
            else:
                return result.data
        else:
            error_msg = f"Failed to process PDF: {result.errors}"
            logger.error(error_msg)
            if return_json:
                return json.dumps({"error": error_msg, "details": result.errors}, indent=2)
            else:
                return {"error": error_msg, "details": result.errors}
    
    def process_batch(
        self,
        input_dir: str,
        file_pattern: str = "*.pdf",
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Process multiple PDF files in batch.
        
        Args:
            input_dir: Directory containing PDF files
            file_pattern: File pattern to match
            progress_callback: Optional progress callback function
            
        Returns:
            BatchResult containing processing results
        """
        logger.info(f"Processing batch from directory: {input_dir}")
        
        return self.batch_processor.process_directory(
            input_dir=input_dir,
            file_pattern=file_pattern,
            progress_callback=progress_callback
        )
    
    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze a PDF to determine its type and recommended processing method.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing PDF: {pdf_path}")
        return detect_pdf_type(pdf_path)
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """
        Get information about supported PDF formats and processing methods.
        
        Returns:
            Information about supported formats
        """
        return {
            "supported_formats": ["PDF"],
            "extraction_methods": [
                {
                    "name": "text_extraction",
                    "description": "Extract text directly from PDFs with selectable text",
                    "best_for": "PDFs created digitally with embedded text"
                },
                {
                    "name": "ocr",
                    "description": "Use OCR to extract text from image-based PDFs",
                    "best_for": "Scanned documents and image-based PDFs",
                    "engines": ["Tesseract", "PaddleOCR"]
                },
                {
                    "name": "hybrid",
                    "description": "Combine text extraction and OCR for best results",
                    "best_for": "PDFs with mixed content or when quality is uncertain"
                }
            ],
            "output_format": "JSON",
            "features": [
                "Automatic PDF type detection",
                "Intelligent fallback mechanisms",
                "Batch processing support",
                "Layout preservation",
                "Confidence scoring",
                "Error handling and recovery"
            ]
        }
    
    def _save_result_to_file(self, data: Dict[str, Any], output_file: str) -> None:
        """
        Save processing result to a file.
        
        Args:
            data: Data to save
            output_file: Output file path
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Result saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving result to file: {e}")


def create_progress_callback():
    """Create a simple progress callback for batch processing."""
    def progress_callback(completed: int, total: int):
        percentage = (completed / total) * 100
        print(f"Progress: {completed}/{total} ({percentage:.1f}%)")
    return progress_callback


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Robust PDF Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single PDF (JSON output only, no PNG images)
  python robust_pdf_processor.py invoice.pdf

  # Process with custom output file
  python robust_pdf_processor.py invoice.pdf --output result.json

  # Process batch of PDFs
  python robust_pdf_processor.py --batch ./invoices/

  # Process and save intermediate PNG images for debugging
  python robust_pdf_processor.py invoice.pdf --save-images

  # Analyze PDF type without processing
  python robust_pdf_processor.py invoice.pdf --analyze-only

  # Get system information
  python robust_pdf_processor.py --info
        """
    )
    
    # Main arguments
    parser.add_argument('pdf_path', nargs='?', help='Path to PDF file to process')
    parser.add_argument('--batch', help='Process all PDFs in directory')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--output-dir', default='./output', help='Output directory')
    
    # Processing options
    parser.add_argument('--confidence-threshold', type=float, default=70.0,
                       help='Minimum confidence threshold (default: 70.0)')
    parser.add_argument('--disable-hybrid', action='store_true',
                       help='Disable hybrid processing mode')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum workers for batch processing (default: 4)')
    parser.add_argument('--save-images', action='store_true',
                       help='Save intermediate PNG images during OCR processing (default: False)')
    
    # Output options
    parser.add_argument('--json-only', action='store_true',
                       help='Output only JSON to stdout')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze PDF type, do not process')
    parser.add_argument('--info', action='store_true',
                       help='Show system information and supported formats')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress all output except results')
    
    args = parser.parse_args()
    
    # Configure logging based on arguments
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize processor
    processor = RobustPDFProcessor(
        output_dir=args.output_dir,
        min_confidence_threshold=args.confidence_threshold,
        enable_hybrid_mode=not args.disable_hybrid,
        max_workers=args.max_workers,
        save_images=args.save_images
    )
    
    try:
        # Handle different modes
        if args.info:
            # Show system information
            info = processor.get_supported_formats()
            print(json.dumps(info, indent=2))
            return 0
        
        elif args.batch:
            # Batch processing mode
            if not Path(args.batch).exists():
                print(f"Error: Directory not found: {args.batch}", file=sys.stderr)
                return 1
            
            progress_callback = None if args.quiet else create_progress_callback()
            batch_result = processor.process_batch(args.batch, progress_callback=progress_callback)
            
            if args.json_only:
                print(json.dumps(batch_result.summary, indent=2))
            else:
                print(f"Batch processing completed:")
                print(f"  Total files: {batch_result.total_files}")
                print(f"  Successful: {batch_result.successful}")
                print(f"  Failed: {batch_result.failed}")
                print(f"  Processing time: {batch_result.processing_time:.2f}s")
            
            return 0 if batch_result.failed == 0 else 1
        
        elif args.pdf_path:
            # Single file processing mode
            if not Path(args.pdf_path).exists():
                print(f"Error: PDF file not found: {args.pdf_path}", file=sys.stderr)
                return 1
            
            if args.analyze_only:
                # Analysis mode
                analysis = processor.analyze_pdf(args.pdf_path)
                print(json.dumps(analysis, indent=2))
                return 0
            else:
                # Processing mode
                result = processor.process_single_pdf(
                    args.pdf_path,
                    output_file=args.output,
                    return_json=args.json_only
                )
                
                if args.json_only:
                    print(result)
                else:
                    if isinstance(result, dict) and 'error' not in result:
                        print(f"Successfully processed: {args.pdf_path}")
                        if args.output:
                            print(f"Result saved to: {args.output}")
                        else:
                            print("Result:")
                            print(json.dumps(result, indent=2))
                    else:
                        print(f"Processing failed: {result.get('error', 'Unknown error')}")
                        return 1
                
                return 0
        
        else:
            # No arguments provided
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
