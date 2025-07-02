"""
Automatic Detection and Fallback Logic

This module implements intelligent switching between extraction methods,
handles edge cases for partially selectable PDFs, and provides comprehensive error handling.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from enum import Enum

from pdf_text_detector import PDFTextDetector
from text_pdf_processor import TextPDFProcessor
from image_pdf_processor import ImagePDFProcessor
from json_schema import UnifiedJSONSchema, create_unified_output
from data_validator import DataValidator
from vendor_detector import VendorDetector, VendorType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """Enumeration of extraction methods."""
    TEXT_EXTRACTION = "text_extraction"
    OCR = "ocr"
    HYBRID = "hybrid"
    TEXT_WITH_OCR_FALLBACK = "text_with_ocr_fallback"
    OCR_WITH_TEXT_FALLBACK = "ocr_with_text_fallback"


class ProcessingResult:
    """Container for processing results."""
    
    def __init__(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        method_used: Optional[ExtractionMethod] = None,
        confidence_score: Optional[float] = None,
        processing_time: Optional[float] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None
    ):
        self.success = success
        self.data = data or {}
        self.method_used = method_used
        self.confidence_score = confidence_score
        self.processing_time = processing_time
        self.errors = errors or []
        self.warnings = warnings or []


class FallbackProcessor:
    """
    Intelligent PDF processor with automatic detection and fallback mechanisms.
    """
    
    def __init__(
        self,
        output_dir: str = "./output",
        min_confidence_threshold: float = 70.0,
        enable_hybrid_mode: bool = True,
        save_images: bool = False,
        **kwargs
    ):
        """
        Initialize the fallback processor.

        Args:
            output_dir: Directory for output files
            min_confidence_threshold: Minimum confidence score to accept results
            enable_hybrid_mode: Whether to enable hybrid processing for mixed content
            save_images: Whether to save intermediate PNG images during OCR processing
            **kwargs: Additional configuration options
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.min_confidence_threshold = min_confidence_threshold
        self.enable_hybrid_mode = enable_hybrid_mode
        self.save_images = save_images

        # Initialize components
        self.detector = PDFTextDetector()
        self.vendor_detector = VendorDetector()
        self.text_processor = TextPDFProcessor()
        self.image_processor = ImagePDFProcessor(
            output_dir=str(self.output_dir),
            save_images=save_images,
            **kwargs
        )
        self.schema = UnifiedJSONSchema()
        self.validator = DataValidator()

        logger.info("FallbackProcessor initialized")
    
    def process_pdf(self, pdf_path: Path) -> ProcessingResult:
        """
        Process a PDF with automatic method detection and fallback.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessingResult containing the extraction results
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        logger.info(f"Starting processing of {pdf_path.name}")
        
        try:
            # Step 1: Detect vendor type first (for routing and specialized processing)
            vendor_result = self.vendor_detector.detect_vendor(pdf_path)
            logger.info(f"Vendor detection: {vendor_result['vendor_name']} "
                       f"(confidence: {vendor_result['confidence']:.2f})")

            # Step 2: Detect PDF type and recommend method
            detection_result = self.detector.detect_pdf_type(str(pdf_path))
            recommended_method = detection_result['recommended_method']

            logger.info(f"Detection result: {detection_result['pdf_type']}, "
                       f"recommended method: {recommended_method}")

            # Step 3: Process based on detection results
            result = self._process_with_method(pdf_path, detection_result, vendor_result)
            
            # Step 3: Validate data quality and potentially apply fallback
            if result.success and result.data:
                validation_result = self.validator.validate_extracted_data(result.data)
                logger.info(f"Data validation - Completeness: {validation_result.completeness_score:.1f}%, "
                           f"Valid: {validation_result.is_valid}")

                if validation_result.missing_fields:
                    logger.warning(f"Missing fields: {validation_result.missing_fields}")

                # Apply fallback if validation fails or confidence is low
                if (not validation_result.is_valid or
                    (result.confidence_score and result.confidence_score < self.min_confidence_threshold)):
                    logger.warning(f"Primary method failed validation or low confidence. Attempting fallback...")
                    result = self._apply_fallback(pdf_path, detection_result, result, vendor_result)
                else:
                    # Enhance the data to fill missing fields
                    result.data = self.validator.enhance_extracted_data(result.data)
                    logger.info("Data enhanced with calculated fields")
            else:
                logger.warning(f"Primary method failed. Attempting fallback...")
                result = self._apply_fallback(pdf_path, detection_result, result, vendor_result)
            
            # Step 4: Finalize result
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            if result.success:
                # Ensure unified schema and add vendor information
                result.data = self._ensure_unified_schema(result.data, str(pdf_path), result, vendor_result)
                logger.info(f"Processing completed successfully in {processing_time:.2f}s")
            else:
                logger.error(f"Processing failed after {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Critical error during processing: {e}")
            return ProcessingResult(
                success=False,
                errors=[f"Critical processing error: {str(e)}"],
                processing_time=processing_time
            )
    
    def _process_with_method(
        self,
        pdf_path: Path,
        detection_result: Dict[str, Any],
        vendor_result: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process PDF with the recommended method.

        Args:
            pdf_path: Path to the PDF file
            detection_result: Results from PDF detection
            vendor_result: Results from vendor detection

        Returns:
            ProcessingResult
        """
        recommended_method = detection_result['recommended_method']

        # Check if we should use specialized processors
        if (vendor_result and
            vendor_result.get('vendor_type') == VendorType.SCHLAGE and
            vendor_result.get('processor_type') == 'schlage_specialized'):
            return self._process_with_schlage_processor(pdf_path, vendor_result)

        if (vendor_result and
            vendor_result.get('vendor_type') == VendorType.STEELCRAFT and
            vendor_result.get('processor_type') == 'steelcraft_specialized'):
            return self._process_with_steelcraft_processor(pdf_path, vendor_result)

        if (vendor_result and
            vendor_result.get('vendor_type') == VendorType.CECO and
            vendor_result.get('processor_type') == 'ceco_specialized'):
            return self._process_with_ceco_processor(pdf_path, vendor_result)

        try:
            if recommended_method == "text_extraction":
                return self._process_with_text_extraction(pdf_path, vendor_result)

            elif recommended_method == "ocr":
                return self._process_with_ocr(pdf_path)

            elif recommended_method == "text_with_ocr_fallback":
                # Try text first, then OCR if needed
                text_result = self._process_with_text_extraction(pdf_path, vendor_result)
                if text_result.success and self._is_result_adequate(text_result):
                    return text_result
                else:
                    logger.info("Text extraction inadequate, falling back to OCR")
                    return self._process_with_ocr(pdf_path)

            elif recommended_method == "ocr_with_text_fallback":
                # Try OCR first, then text if needed
                ocr_result = self._process_with_ocr(pdf_path)
                if ocr_result.success and self._is_result_adequate(ocr_result):
                    return ocr_result
                else:
                    logger.info("OCR inadequate, falling back to text extraction")
                    # Don't save raw text for fallback methods
                    return self._process_with_text_extraction(pdf_path, vendor_result, save_raw_text=False)

            elif self.enable_hybrid_mode and detection_result.get('mixed_content', False):
                return self._process_with_hybrid_method(pdf_path, vendor_result)

            else:
                # Default to text extraction
                return self._process_with_text_extraction(pdf_path, vendor_result)
                
        except Exception as e:
            logger.error(f"Error in method-specific processing: {e}")
            return ProcessingResult(
                success=False,
                errors=[f"Method processing error: {str(e)}"]
            )
    
    def _process_with_text_extraction(self, pdf_path: Path, vendor_result: Optional[Dict[str, Any]] = None, save_raw_text: bool = True) -> ProcessingResult:
        """Process PDF using text extraction."""
        try:
            logger.info("Processing with text extraction")

            # Determine vendor folder for raw text saving
            vendor_folder = None
            if save_raw_text and vendor_result and vendor_result.get('vendor_type'):
                vendor_type = vendor_result['vendor_type']
                if hasattr(vendor_type, 'value'):
                    vendor_folder = vendor_type.value.lower()
                else:
                    vendor_folder = str(vendor_type).lower()

            data = self.text_processor.process_pdf(str(pdf_path), vendor_folder=vendor_folder if save_raw_text else None)

            # Calculate confidence based on extracted content
            confidence = self._calculate_text_confidence(data)

            return ProcessingResult(
                success=True,
                data=data,
                method_used=ExtractionMethod.TEXT_EXTRACTION,
                confidence_score=confidence
            )

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ProcessingResult(
                success=False,
                errors=[f"Text extraction error: {str(e)}"]
            )
    
    def _process_with_ocr(self, pdf_path: Path) -> ProcessingResult:
        """Process PDF using OCR."""
        try:
            logger.info("Processing with OCR")
            data = self.image_processor.process_pdf(str(pdf_path))
            
            # Extract confidence from OCR metadata
            confidence = self._extract_ocr_confidence(data)
            
            return ProcessingResult(
                success=True,
                data=data,
                method_used=ExtractionMethod.OCR,
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return ProcessingResult(
                success=False,
                errors=[f"OCR processing error: {str(e)}"]
            )
    
    def _process_with_hybrid_method(self, pdf_path: Path, vendor_result: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process PDF using hybrid method (combination of text and OCR)."""
        try:
            logger.info("Processing with hybrid method")

            # Try both methods
            text_result = self._process_with_text_extraction(pdf_path, vendor_result)
            ocr_result = self._process_with_ocr(pdf_path)
            
            # Combine results intelligently
            combined_data = self._combine_extraction_results(
                text_result.data if text_result.success else {},
                ocr_result.data if ocr_result.success else {}
            )
            
            # Calculate combined confidence
            text_conf = text_result.confidence_score or 0
            ocr_conf = ocr_result.confidence_score or 0
            combined_confidence = (text_conf + ocr_conf) / 2
            
            return ProcessingResult(
                success=True,
                data=combined_data,
                method_used=ExtractionMethod.HYBRID,
                confidence_score=combined_confidence,
                warnings=text_result.warnings + ocr_result.warnings
            )
            
        except Exception as e:
            logger.error(f"Hybrid processing failed: {e}")
            return ProcessingResult(
                success=False,
                errors=[f"Hybrid processing error: {str(e)}"]
            )
    
    def _apply_fallback(
        self,
        pdf_path: Path,
        detection_result: Dict[str, Any],
        primary_result: ProcessingResult,
        vendor_result: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Apply fallback processing when primary method fails or has low confidence.

        Args:
            pdf_path: Path to the PDF file
            detection_result: Original detection results
            primary_result: Result from primary processing attempt
            vendor_result: Results from vendor detection

        Returns:
            ProcessingResult from fallback attempt
        """
        logger.info("Applying fallback processing")

        # Determine fallback method
        if primary_result.method_used == ExtractionMethod.TEXT_EXTRACTION:
            fallback_result = self._process_with_ocr(pdf_path)
        elif primary_result.method_used == ExtractionMethod.OCR:
            # Don't save raw text for fallback methods
            fallback_result = self._process_with_text_extraction(pdf_path, vendor_result, save_raw_text=False)
        else:
            # If hybrid failed, try the opposite of what was recommended
            if detection_result['recommended_method'] == "text_extraction":
                fallback_result = self._process_with_ocr(pdf_path)
            else:
                # Don't save raw text for fallback methods
                fallback_result = self._process_with_text_extraction(pdf_path, vendor_result, save_raw_text=False)
        
        # Compare results and return the better one
        if fallback_result.success:
            if (fallback_result.confidence_score or 0) > (primary_result.confidence_score or 0):
                logger.info("Fallback method produced better results")
                return fallback_result
            elif primary_result.success:
                logger.info("Primary method results retained despite low confidence")
                return primary_result
            else:
                logger.info("Using fallback results as primary failed")
                return fallback_result
        else:
            logger.warning("Fallback method also failed")
            return primary_result
    
    def _is_result_adequate(self, result: ProcessingResult) -> bool:
        """Check if processing result is adequate using comprehensive validation."""
        if not result.success:
            return False

        if result.confidence_score and result.confidence_score < self.min_confidence_threshold:
            return False

        # Check if essential data is present
        data = result.data
        if not data:
            return False

        # Use the data validator for comprehensive adequacy check
        validation_result = self.validator.validate_extracted_data(data)

        # Consider result adequate if:
        # 1. Validation passes OR
        # 2. Completeness score is above 70% with minimal missing critical fields
        is_adequate = (
            validation_result.is_valid or
            (validation_result.completeness_score >= 70.0 and
             len([f for f in validation_result.missing_fields
                  if any(critical in f for critical in ['invoice_number', 'line_items', 'company_name'])]) <= 1)
        )

        if not is_adequate:
            logger.info(f"Result not adequate - Completeness: {validation_result.completeness_score:.1f}%, "
                       f"Missing: {validation_result.missing_fields}")

        return is_adequate
    
    def _calculate_text_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score for text extraction results."""
        confidence = 0.0

        # Check for presence and quality of key fields
        invoice_header = data.get('invoice_header', {})
        if invoice_header.get('invoice_number') and str(invoice_header['invoice_number']).strip():
            confidence += 15
        if invoice_header.get('invoice_date') and str(invoice_header['invoice_date']).strip():
            confidence += 10

        vendor_info = data.get('vendor_info', {})
        if vendor_info.get('company_name') and str(vendor_info['company_name']).strip():
            confidence += 10

        customer_info = data.get('customer_info', {})
        if customer_info.get('company_name') and str(customer_info['company_name']).strip():
            confidence += 10

        # Line items are critical - give more weight and check quality
        line_items = data.get('line_items', [])
        if line_items:
            base_line_items_score = 30

            # Quality bonus for line items
            quality_bonus = 0
            valid_items = 0

            for item in line_items:
                if self._is_complete_line_item(item):
                    valid_items += 1
                    quality_bonus += 2  # 2 points per complete item

            # Cap quality bonus at 25 points
            quality_bonus = min(quality_bonus, 25)
            confidence += base_line_items_score + quality_bonus

            # Penalty if too few valid items
            if valid_items < len(line_items) * 0.7:  # Less than 70% valid
                confidence -= 10

        # Totals and other fields
        totals = data.get('totals', {})
        if totals and any(v for v in totals.values() if str(v).strip()):
            confidence += 8

        payment_terms = data.get('payment_terms', {})
        if payment_terms and any(v for v in payment_terms.values() if str(v).strip()):
            confidence += 5

        shipping_info = data.get('shipping_info', {})
        if shipping_info and any(v for v in shipping_info.values() if str(v).strip()):
            confidence += 5

        return min(confidence, 100.0)

    def _is_complete_line_item(self, item: Dict[str, Any]) -> bool:
        """Check if a line item has all essential fields filled."""
        required_fields = ['line_number', 'description']
        numeric_fields = ['quantity_ordered', 'unit_price', 'extended_amount']

        # Check required text fields
        for field in required_fields:
            if not item.get(field) or str(item[field]).strip() == "":
                return False

        # Check at least 2 numeric fields have valid values
        valid_numeric = sum(
            1 for field in numeric_fields
            if isinstance(item.get(field), (int, float)) and item.get(field, 0) > 0
        )

        return valid_numeric >= 2
    
    def _extract_ocr_confidence(self, data: Dict[str, Any]) -> float:
        """Extract confidence score from OCR metadata."""
        ocr_details = data.get('ocr_details', {})
        pages = ocr_details.get('pages', [])
        
        if not pages:
            return 0.0
        
        # Average confidence across all pages
        total_confidence = 0.0
        valid_pages = 0
        
        for page in pages:
            page_confidence = page.get('confidence_score', 0)
            if page_confidence > 0:
                total_confidence += page_confidence
                valid_pages += 1
        
        return total_confidence / valid_pages if valid_pages > 0 else 0.0

    def _combine_extraction_results(
        self,
        text_data: Dict[str, Any],
        ocr_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Intelligently combine results from text extraction and OCR.

        Args:
            text_data: Results from text extraction
            ocr_data: Results from OCR

        Returns:
            Combined data dictionary
        """
        combined = {}

        # Define priority for each section (text_extraction vs ocr)
        section_priorities = {
            'invoice_header': 'text',  # Text extraction usually better for headers
            'vendor_info': 'text',
            'customer_info': 'text',
            'line_items': 'text',      # Tables better extracted via text
            'totals': 'ocr',           # OCR might catch totals better
            'payment_terms': 'text',
            'shipping_info': 'ocr'     # OCR might be better for shipping labels
        }

        for section in section_priorities:
            primary_source = text_data if section_priorities[section] == 'text' else ocr_data
            secondary_source = ocr_data if section_priorities[section] == 'text' else text_data

            # Use primary source if it has data, otherwise use secondary
            primary_section = primary_source.get(section, {})
            secondary_section = secondary_source.get(section, {})

            if self._has_meaningful_data(primary_section):
                combined[section] = primary_section
            elif self._has_meaningful_data(secondary_section):
                combined[section] = secondary_section
            else:
                # Merge both if neither is clearly better
                combined[section] = self._merge_sections(primary_section, secondary_section)

        # Combine metadata
        combined['metadata'] = self._merge_metadata(
            text_data.get('metadata', {}),
            ocr_data.get('metadata', {})
        )

        return combined

    def _has_meaningful_data(self, section: Any) -> bool:
        """Check if a section contains meaningful data."""
        if not section:
            return False

        if isinstance(section, dict):
            # Check if dict has non-empty values
            return any(value for value in section.values() if value)
        elif isinstance(section, list):
            # Check if list has items
            return len(section) > 0
        else:
            return bool(section)

    def _merge_sections(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two sections, preferring non-empty values from primary."""
        merged = secondary.copy() if secondary else {}

        if primary:
            for key, value in primary.items():
                if value:  # Only use non-empty values from primary
                    merged[key] = value

        return merged

    def _merge_metadata(self, text_meta: Dict[str, Any], ocr_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Merge metadata from both extraction methods."""
        merged_meta = {
            'extraction_method': 'hybrid',
            'processor': 'FallbackProcessor',
            'methods_used': []
        }

        # Combine information from both sources
        if text_meta:
            merged_meta.update(text_meta)
            merged_meta['methods_used'].append('text_extraction')

        if ocr_meta:
            # Don't overwrite with OCR metadata, but add OCR-specific info
            merged_meta['ocr_engines_used'] = ocr_meta.get('ocr_engines_used', [])
            merged_meta['methods_used'].append('ocr')

            # Use higher page count if available
            if 'total_pages' in ocr_meta:
                merged_meta['total_pages'] = max(
                    merged_meta.get('total_pages', 0),
                    ocr_meta['total_pages']
                )

        return merged_meta

    def _ensure_unified_schema(
        self,
        data: Dict[str, Any],
        pdf_path: str,
        result: ProcessingResult,
        vendor_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ensure the output follows the unified schema.

        Args:
            data: Extracted data
            pdf_path: Path to the PDF file
            result: Processing result

        Returns:
            Data conforming to unified schema
        """
        # Update metadata with processing information
        metadata = data.get('metadata', {})
        metadata.update({
            'pdf_path': str(pdf_path),
            'confidence_score': result.confidence_score,
            'processing_time_seconds': result.processing_time
        })

        # Add vendor information to metadata if available (temporarily disabled to fix schema issues)
        # if vendor_result:
        #     metadata.update({
        #         'vendor_type': vendor_result.get('vendor_name', 'unknown'),
        #         'vendor_confidence': vendor_result.get('confidence', 0.0),
        #         'vendor_detection_method': vendor_result.get('detection_method', 'unknown'),
        #         'vendor_folder': vendor_result.get('output_folder', 'output/unknown')
        #     })

        # Create unified output - bypass dataclass issues for now
        unified_data = {
            'invoice_header': data.get('invoice_header', {}),
            'vendor_info': data.get('vendor_info', {}),
            'customer_info': data.get('customer_info', {}),
            'line_items': data.get('line_items', []),
            'totals': data.get('totals', {}),
            'payment_terms': data.get('payment_terms', {}),
            'shipping_info': data.get('shipping_info', {}),
            'metadata': metadata
        }

        # Add any additional fields that might be present
        for key, value in data.items():
            if key not in unified_data:
                unified_data[key] = value

        return unified_data

    def _process_with_steelcraft_processor(self, pdf_path: Path, vendor_result: Dict[str, Any]) -> ProcessingResult:
        """
        Process Steelcraft PDFs using the specialized Steelcraft processor.

        Args:
            pdf_path: Path to the PDF file
            vendor_result: Vendor detection results

        Returns:
            ProcessingResult
        """
        try:
            from steelcraft_processor import SteelcraftInvoiceProcessor

            # Get the vendor-specific output directory
            output_folder = vendor_result.get('output_folder', 'output/steelcraft')

            # Initialize Steelcraft processor
            steelcraft_processor = SteelcraftInvoiceProcessor(output_dir=output_folder)

            # Process the PDF
            result_data = steelcraft_processor.process_steelcraft_pdf(str(pdf_path))

            # Convert Steelcraft data to compatible format
            compatible_data = self._convert_steelcraft_data_to_standard(result_data)

            return ProcessingResult(
                success=True,
                data=compatible_data,
                method_used=ExtractionMethod.TEXT_EXTRACTION,  # Steelcraft uses text extraction
                confidence_score=95.0,  # High confidence for specialized processor
                processing_time=0.0,
                errors=[]
            )

        except Exception as e:
            logger.error(f"Steelcraft processor failed: {e}")
            return ProcessingResult(
                success=False,
                errors=[f"Steelcraft processor error: {str(e)}"]
            )

    def _process_with_schlage_processor(self, pdf_path: Path, vendor_result: Dict[str, Any]) -> ProcessingResult:
        """
        Process Schlage PDFs using the specialized Schlage processor.

        Args:
            pdf_path: Path to the PDF file
            vendor_result: Vendor detection results

        Returns:
            ProcessingResult
        """
        try:
            from schlage_processor import SchlageInvoiceProcessor

            # Get the vendor-specific output directory
            output_folder = vendor_result.get('output_folder', 'output/schlage')

            # Initialize Schlage processor
            schlage_processor = SchlageInvoiceProcessor(output_dir=output_folder)

            # Process the PDF
            result_data = schlage_processor.process_schlage_pdf(str(pdf_path))

            # Convert Schlage data to compatible format
            compatible_data = self._convert_schlage_data_to_standard(result_data)

            return ProcessingResult(
                success=True,
                data=compatible_data,
                method_used=ExtractionMethod.OCR,  # Schlage processor uses OCR
                confidence_score=result_data.get('metadata', {}).get('accuracy_score', 0.0),
                processing_time=0.0,  # Will be calculated by caller
                errors=[]
            )

        except Exception as e:
            logger.error(f"Schlage processor failed: {e}")
            return ProcessingResult(
                success=False,
                errors=[f"Schlage processor error: {str(e)}"]
            )

    def _process_with_ceco_processor(self, pdf_path: Path, vendor_result: Dict[str, Any]) -> ProcessingResult:
        """
        Process CECO PDFs using the specialized CECO processor.

        Args:
            pdf_path: Path to the PDF file
            vendor_result: Vendor detection results

        Returns:
            ProcessingResult
        """
        try:
            from ceco_processor import CECOInvoiceProcessor

            # Get the vendor-specific output directory
            output_folder = vendor_result.get('output_folder', 'output/ceco')

            # Initialize CECO processor
            ceco_processor = CECOInvoiceProcessor(output_dir=output_folder)

            # Process the PDF
            result_data = ceco_processor.process_ceco_pdf(str(pdf_path))

            # Convert CECO data to compatible format
            compatible_data = self._convert_ceco_data_to_standard(result_data)

            return ProcessingResult(
                success=True,
                data=compatible_data,
                method_used=ExtractionMethod.TEXT_EXTRACTION,  # CECO processor uses text extraction
                confidence_score=result_data.get('metadata', {}).get('confidence_score', 85.0),  # Dynamic confidence from processor
                processing_time=0.0,
                errors=[]
            )

        except Exception as e:
            logger.error(f"CECO processor failed: {e}")
            return ProcessingResult(
                success=False,
                errors=[f"CECO processor error: {str(e)}"]
            )

    def _convert_steelcraft_data_to_standard(self, steelcraft_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Steelcraft processor output to standard schema format.

        Args:
            steelcraft_data: Data from Steelcraft processor

        Returns:
            Data compatible with standard schema
        """
        # Steelcraft data is already in standard format, just return it
        return steelcraft_data

    def _convert_ceco_data_to_standard(self, ceco_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert CECO processor output to standard schema format.

        Args:
            ceco_data: Data from CECO processor

        Returns:
            Data compatible with standard schema
        """
        # CECO data is already in standard format, just return it
        return ceco_data

    def _convert_schlage_data_to_standard(self, schlage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Schlage processor output to standard schema format.

        Args:
            schlage_data: Data from Schlage processor

        Returns:
            Data compatible with standard schema
        """
        # Create a copy to avoid modifying original
        standard_data = {}

        # Convert invoice header - filter out unsupported fields
        if 'invoice_header' in schlage_data:
            header = schlage_data['invoice_header']
            standard_data['invoice_header'] = {
                'invoice_number': header.get('invoice_number'),
                'invoice_date': header.get('invoice_date'),
                'order_number': header.get('order_number'),
                'customer_po': header.get('customer_po'),
                'due_date': header.get('due_date')
                # Skip 'order_date' as it's not in standard schema
            }

        # Convert vendor info - flatten nested structures
        if 'vendor_info' in schlage_data:
            vendor = schlage_data['vendor_info']
            standard_data['vendor_info'] = {
                'company_name': vendor.get('company_name'),
                'address': vendor.get('address'),
                'city_state_zip': vendor.get('city_state_zip'),
                'phone': vendor.get('commercial_support'),  # Use commercial support as main phone
                'email': vendor.get('email')
                # Skip nested 'remit_to' structure
            }

        # Convert customer info - preserve bill_to and ship_to structures
        if 'customer_info' in schlage_data:
            customer = schlage_data['customer_info']
            # Preserve the full customer_info structure including bill_to and ship_to
            standard_data['customer_info'] = customer.copy()

        # Copy other sections as-is if they're compatible
        for section in ['line_items', 'totals', 'payment_terms', 'shipping_info', 'metadata']:
            if section in schlage_data:
                standard_data[section] = schlage_data[section]

        return standard_data


def process_pdf_with_fallback(pdf_path: str, **kwargs) -> ProcessingResult:
    """
    Convenience function to process a PDF with automatic fallback.

    Args:
        pdf_path: Path to the PDF file
        **kwargs: Additional arguments for FallbackProcessor

    Returns:
        ProcessingResult
    """
    processor = FallbackProcessor(**kwargs)
    return processor.process_pdf(Path(pdf_path))
