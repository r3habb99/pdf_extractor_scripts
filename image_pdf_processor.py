"""
Image-based PDF Processor

This module handles extraction from image-based (scanned) PDFs using multiple OCR engines
with advanced preprocessing and accuracy enhancement techniques.
"""

import os
import cv2
import numpy as np
import pytesseract
from pdf2image.pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import re

# Import enhanced pattern detector
from enhanced_pattern_detector import DynamicPatternDetector, FieldType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("PaddleOCR not available. Only Tesseract will be used.")


class ImagePDFProcessor:
    """
    Processes image-based PDFs using multiple OCR engines with enhanced preprocessing.
    """
    
    def __init__(self, output_dir: str = "./output", use_paddleocr: bool = False, save_images: bool = False):
        """
        Initialize the enhanced image PDF processor.

        Args:
            output_dir: Directory to save intermediate images
            use_paddleocr: Whether to use PaddleOCR in addition to Tesseract
            save_images: Whether to save intermediate PNG images (default: False)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.save_images = save_images

        # Initialize enhanced pattern detector
        self.pattern_detector = DynamicPatternDetector()
        logger.info("Enhanced dynamic pattern detector initialized")

        # Initialize OCR engines
        self.use_paddleocr = use_paddleocr and PADDLEOCR_AVAILABLE
        if self.use_paddleocr:
            try:
                logger.info("Initializing PaddleOCR...")
                self.paddle_ocr = PaddleOCR(lang='en', use_angle_cls=True)
                logger.info("PaddleOCR initialized successfully!")
            except Exception as e:
                logger.error(f"Error initializing PaddleOCR: {e}")
                self.use_paddleocr = False

        # Enhanced Tesseract configuration for better invoice processing
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/$%()-:/ '
    
    def process_pdf(self, pdf_path: 'str | Path') -> Dict[str, Any]:
        """
        Process an image-based PDF and extract structured data.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Structured data dictionary maintaining PDF layout
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing image-based PDF: {pdf_path.name}")
        
        # Convert PDF to images
        images = self._convert_pdf_to_images(pdf_path)
        
        # Process each page with OCR
        all_ocr_results = []
        combined_text = ""
        
        for idx, img in enumerate(images):
            page_num = idx + 1
            logger.info(f"Processing page {page_num}/{len(images)}")

            # Save original image only if save_images is enabled
            if self.save_images:
                image_path = self.output_dir / f"{pdf_path.stem}_page{page_num}.png"
                img.save(image_path)
            else:
                # Create temporary image path for OCR processing without saving
                image_path = self.output_dir / f"temp_{pdf_path.stem}_page{page_num}.png"
                img.save(image_path)

            # Apply OCR with multiple engines
            ocr_result = self._apply_multi_engine_ocr(image_path, page_num)
            all_ocr_results.append(ocr_result)
            combined_text += ocr_result['best_text'] + "\n"

            # Clean up temporary image if not saving images
            if not self.save_images and image_path.exists():
                image_path.unlink()
        
        # Extract structured data from combined OCR results
        structured_data = self._extract_structured_data(combined_text, all_ocr_results)
        
        # Add metadata
        structured_data['metadata'] = {
            'pdf_path': str(pdf_path),
            'extraction_method': 'ocr',
            'total_pages': len(images),
            'processor': 'ImagePDFProcessor',
            'ocr_engines_used': self._get_engines_used()
        }
        
        logger.info(f"OCR processing complete for {pdf_path.name}")
        return structured_data
    
    def _convert_pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """
        Convert PDF to high-quality images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PIL Image objects
        """
        try:
            # Use high DPI for better OCR accuracy
            images = convert_from_path(pdf_path, dpi=300, fmt='PNG')
            logger.info(f"Converted PDF to {len(images)} images")
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
    
    def _apply_multi_engine_ocr(self, image_path: Path, page_num: int) -> Dict[str, Any]:
        """
        Apply multiple OCR engines to an image and combine results.
        
        Args:
            image_path: Path to the image file
            page_num: Page number
            
        Returns:
            Dictionary containing OCR results from multiple engines
        """
        ocr_results = {
            'page_number': page_num,
            'image_path': str(image_path),
            'tesseract_result': {},
            'paddleocr_result': {},
            'best_text': "",
            'confidence_score': 0.0
        }
        
        # Preprocess image for better OCR
        preprocessed_images = self._preprocess_image(image_path)
        
        # Apply Tesseract OCR
        tesseract_result = self._apply_tesseract_ocr(preprocessed_images)
        ocr_results['tesseract_result'] = tesseract_result
        
        # Apply PaddleOCR if available
        if self.use_paddleocr:
            paddleocr_result = self._apply_paddleocr(image_path)
            ocr_results['paddleocr_result'] = paddleocr_result
        
        # Determine best result
        ocr_results['best_text'], ocr_results['confidence_score'] = self._select_best_ocr_result(
            tesseract_result, ocr_results.get('paddleocr_result', {})
        )

        # Clean up temporary preprocessed images if not saving images
        if not self.save_images and '_temp_files' in preprocessed_images:
            temp_files_list = preprocessed_images['_temp_files']
            if isinstance(temp_files_list, list):
                for temp_file in temp_files_list:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception as e:
                        logger.warning(f"Could not delete temporary file {temp_file}: {e}")

        return ocr_results
    
    def _preprocess_image(self, image_path: Path) -> Dict[str, Path]:
        """
        Apply advanced preprocessing techniques to improve OCR accuracy.
        Enhanced with multiple preprocessing strategies for different document types.

        Args:
            image_path: Path to the original image

        Returns:
            Dictionary of preprocessed image paths with quality scores
        """
        preprocessed_images = {}
        temp_files = []  # Track temporary files for cleanup

        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return {image_path.stem: image_path}

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Enhanced preprocessing pipeline
            preprocessing_methods = self._get_enhanced_preprocessing_methods(gray, image_path, temp_files)

            for method_name, processed_img in preprocessing_methods.items():
                if processed_img is not None:
                    method_path = self._save_preprocessed_image(processed_img, image_path, method_name, temp_files)
                    if method_path:
                        preprocessed_images[method_name] = method_path

        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            preprocessed_images['original'] = image_path

        # Store temp files for later cleanup
        if not self.save_images:
            preprocessed_images['_temp_files'] = temp_files

        return preprocessed_images

    def _get_enhanced_preprocessing_methods(self, gray: np.ndarray, image_path: Path, temp_files: List[Path]) -> Dict[str, np.ndarray]:
        """
        Apply multiple enhanced preprocessing methods for better OCR accuracy.

        Args:
            gray: Grayscale image
            image_path: Original image path
            temp_files: List to track temporary files

        Returns:
            Dictionary of method names to processed images
        """
        methods = {}

        try:
            # 1. Basic OTSU threshold (baseline)
            _, thresh_basic = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            methods['basic_otsu'] = thresh_basic

            # 2. Adaptive threshold with multiple parameters
            adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            methods['adaptive_mean'] = adaptive_mean

            adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            methods['adaptive_gaussian'] = adaptive_gaussian

            # 3. Advanced denoising + threshold
            denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
            _, thresh_denoised = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            methods['denoised_otsu'] = thresh_denoised

            # 4. Morphological operations for text enhancement
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            morph_close = cv2.morphologyEx(thresh_basic, cv2.MORPH_CLOSE, kernel)
            methods['morphological'] = morph_close

            # 5. Contrast enhancement + threshold
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh_enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            methods['clahe_enhanced'] = thresh_enhanced

            # 6. Bilateral filter + threshold (preserves edges)
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            _, thresh_bilateral = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            methods['bilateral_filtered'] = thresh_bilateral

            # 7. Gaussian blur + threshold
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh_blurred = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            methods['gaussian_blurred'] = thresh_blurred

            # 8. Erosion + dilation for text cleanup
            kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            eroded = cv2.erode(thresh_basic, kernel_small, iterations=1)
            dilated = cv2.dilate(eroded, kernel_small, iterations=1)
            methods['erode_dilate'] = dilated

        except Exception as e:
            logger.warning(f"Error in enhanced preprocessing: {e}")
            # Fallback to basic threshold
            _, thresh_basic = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            methods['fallback_basic'] = thresh_basic

        return methods

    def _save_preprocessed_image(self, processed_img: np.ndarray, original_path: Path, method_name: str, temp_files: List[Path]) -> Optional[Path]:
        """
        Save preprocessed image to disk.

        Args:
            processed_img: Processed image array
            original_path: Original image path
            method_name: Name of preprocessing method
            temp_files: List to track temporary files

        Returns:
            Path to saved image or None if failed
        """
        try:
            if self.save_images:
                save_path = original_path.parent / f"{original_path.stem}_{method_name}.png"
            else:
                save_path = original_path.parent / f"temp_{original_path.stem}_{method_name}.png"
                temp_files.append(save_path)

            cv2.imwrite(str(save_path), processed_img)
            return save_path

        except Exception as e:
            logger.warning(f"Failed to save preprocessed image {method_name}: {e}")
            return None
    
    def _apply_tesseract_ocr(self, preprocessed_images: Dict[str, Path]) -> Dict[str, Any]:
        """
        Apply Tesseract OCR to preprocessed images.
        
        Args:
            preprocessed_images: Dictionary of preprocessed image paths
            
        Returns:
            Tesseract OCR results
        """
        tesseract_results = {
            'engine': 'tesseract',
            'results': {},
            'best_text': "",
            'best_confidence': 0.0
        }
        
        for variant, image_path in preprocessed_images.items():
            try:
                # Extract text
                text = pytesseract.image_to_string(
                    Image.open(image_path), 
                    config=self.tesseract_config
                )
                
                # Get detailed data with confidence scores
                data = pytesseract.image_to_data(
                    Image.open(image_path), 
                    output_type=pytesseract.Output.DICT,
                    config=self.tesseract_config
                )
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                tesseract_results['results'][variant] = {
                    'text': text,
                    'confidence': avg_confidence,
                    'word_count': len([word for word in text.split() if word.strip()])
                }
                
                # Track best result
                if avg_confidence > tesseract_results['best_confidence']:
                    tesseract_results['best_text'] = text
                    tesseract_results['best_confidence'] = avg_confidence
                    
            except Exception as e:
                logger.warning(f"Tesseract OCR failed for {variant}: {e}")
                continue
        
        return tesseract_results
    
    def _apply_paddleocr(self, image_path: Path) -> Dict[str, Any]:
        """
        Apply PaddleOCR to the image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PaddleOCR results
        """
        paddleocr_result = {
            'engine': 'paddleocr',
            'text': "",
            'confidence': 0.0,
            'word_count': 0
        }
        
        try:
            result = self.paddle_ocr.ocr(str(image_path))
            
            if result and result[0]:
                # Extract text and calculate average confidence
                texts = []
                confidences = []
                
                for line in result[0]:
                    if len(line) >= 2 and line[1]:
                        text, confidence = line[1]
                        texts.append(text)
                        confidences.append(confidence)
                
                full_text = " ".join(texts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                paddleocr_result.update({
                    'text': full_text,
                    'confidence': avg_confidence * 100,  # Convert to percentage
                    'word_count': len(full_text.split())
                })
                
        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}")
        
        return paddleocr_result
    
    def _select_best_ocr_result(
        self,
        tesseract_result: Dict[str, Any],
        paddleocr_result: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Enhanced OCR result selection using multiple quality metrics.

        Args:
            tesseract_result: Results from Tesseract
            paddleocr_result: Results from PaddleOCR

        Returns:
            Tuple of (best_text, confidence_score)
        """
        # Get results from both engines
        tesseract_text = tesseract_result.get('best_text', "")
        tesseract_confidence = tesseract_result.get('best_confidence', 0)

        paddleocr_text = paddleocr_result.get('text', "")
        paddleocr_confidence = paddleocr_result.get('confidence', 0)

        # If only one engine has results, use it
        if not tesseract_text.strip() and paddleocr_text.strip():
            return paddleocr_text, paddleocr_confidence
        elif tesseract_text.strip() and not paddleocr_text.strip():
            return tesseract_text, tesseract_confidence
        elif not tesseract_text.strip() and not paddleocr_text.strip():
            return "", 0.0

        # Both engines have results - use advanced scoring
        tesseract_score = self._calculate_ocr_quality_score(tesseract_text, tesseract_confidence)
        paddleocr_score = self._calculate_ocr_quality_score(paddleocr_text, paddleocr_confidence)

        logger.info(f"OCR Quality Scores - Tesseract: {tesseract_score:.2f}, PaddleOCR: {paddleocr_score:.2f}")

        # Select best result or combine if scores are close
        if abs(tesseract_score - paddleocr_score) < 10:  # Scores are close
            # Try to combine results intelligently
            combined_text = self._combine_ocr_results(tesseract_text, paddleocr_text)
            combined_confidence = max(tesseract_confidence, paddleocr_confidence)
            return combined_text, combined_confidence
        elif paddleocr_score > tesseract_score:
            return paddleocr_text, paddleocr_confidence
        else:
            return tesseract_text, tesseract_confidence
    
    def _calculate_ocr_quality_score(self, text: str, confidence: float) -> float:
        """
        Calculate comprehensive quality score for OCR result.

        Args:
            text: OCR extracted text
            confidence: OCR confidence score

        Returns:
            Quality score (0-100)
        """
        if not text.strip():
            return 0.0

        score = 0.0

        # Base confidence score (40% weight)
        score += confidence * 0.4

        # Text length and word count (20% weight)
        word_count = len(text.split())
        char_count = len(text.strip())
        length_score = min(100, (word_count * 5) + (char_count * 0.5))
        score += length_score * 0.2

        # Invoice-specific content detection (25% weight)
        invoice_keywords = [
            'invoice', 'total', 'amount', 'date', 'number', 'customer', 'vendor',
            'quantity', 'price', 'description', 'line', 'item', 'tax', 'subtotal'
        ]
        keyword_matches = sum(1 for keyword in invoice_keywords if keyword.lower() in text.lower())
        keyword_score = min(100, keyword_matches * 10)
        score += keyword_score * 0.25

        # Numeric content detection (10% weight)
        numeric_patterns = re.findall(r'\d+\.?\d*', text)
        numeric_score = min(100, len(numeric_patterns) * 5)
        score += numeric_score * 0.1

        # Text quality indicators (5% weight)
        quality_indicators = 0
        if re.search(r'\d{2,}', text):  # Has multi-digit numbers
            quality_indicators += 20
        if re.search(r'[A-Z]{2,}', text):  # Has uppercase sequences
            quality_indicators += 20
        if re.search(r'\$\d+', text):  # Has currency amounts
            quality_indicators += 30
        if re.search(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', text):  # Has dates
            quality_indicators += 30

        score += quality_indicators * 0.05

        return min(100, score)

    def _combine_ocr_results(self, text1: str, text2: str) -> str:
        """
        Intelligently combine results from two OCR engines.

        Args:
            text1: Text from first OCR engine
            text2: Text from second OCR engine

        Returns:
            Combined text
        """
        if not text1.strip():
            return text2
        if not text2.strip():
            return text1

        # Split into lines for comparison
        lines1 = [line.strip() for line in text1.split('\n') if line.strip()]
        lines2 = [line.strip() for line in text2.split('\n') if line.strip()]

        combined_lines = []

        # Use longer result as base
        base_lines = lines1 if len(lines1) >= len(lines2) else lines2
        alt_lines = lines2 if len(lines1) >= len(lines2) else lines1

        for i, base_line in enumerate(base_lines):
            if i < len(alt_lines):
                alt_line = alt_lines[i]
                # Choose line with more invoice-relevant content
                if self._line_has_more_invoice_content(alt_line, base_line):
                    combined_lines.append(alt_line)
                else:
                    combined_lines.append(base_line)
            else:
                combined_lines.append(base_line)

        return '\n'.join(combined_lines)

    def _line_has_more_invoice_content(self, line1: str, line2: str) -> bool:
        """
        Determine which line has more relevant invoice content.

        Args:
            line1: First line to compare
            line2: Second line to compare

        Returns:
            True if line1 has more invoice content
        """
        # Count numeric content
        nums1 = len(re.findall(r'\d+\.?\d*', line1))
        nums2 = len(re.findall(r'\d+\.?\d*', line2))

        # Count invoice keywords
        keywords = ['invoice', 'total', 'amount', 'date', 'qty', 'price', 'description']
        keywords1 = sum(1 for kw in keywords if kw.lower() in line1.lower())
        keywords2 = sum(1 for kw in keywords if kw.lower() in line2.lower())

        # Prefer line with more numbers and keywords
        score1 = nums1 * 2 + keywords1 * 3 + len(line1.split())
        score2 = nums2 * 2 + keywords2 * 3 + len(line2.split())

        return score1 > score2

    def _get_engines_used(self) -> List[str]:
        """Get list of OCR engines used."""
        engines = ['tesseract']
        if self.use_paddleocr:
            engines.append('paddleocr')
        return engines

    def _extract_structured_data(self, combined_text: str, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract structured data from OCR text results using enhanced dynamic patterns.

        Args:
            combined_text: Combined text from all pages
            ocr_results: List of OCR results for each page

        Returns:
            Structured data dictionary with confidence scores
        """
        logger.info("Starting enhanced structured data extraction")

        # Initialize structured data with consistent schema
        structured_data = {
            "invoice_header": {},
            "vendor_info": {},
            "customer_info": {},
            "line_items": [],
            "totals": {},
            "payment_terms": {},
            "shipping_info": {},
            "extraction_metadata": {
                "method": "enhanced_dynamic_patterns",
                "confidence_scores": {},
                "patterns_used": {}
            }
        }

        # Use dynamic pattern detector for enhanced extraction
        extracted_fields = self.pattern_detector.extract_all_fields(combined_text)

        # Map extracted fields to structured data with confidence tracking
        self._map_extracted_fields_to_structure(extracted_fields, structured_data)

        # Extract line items using enhanced methods
        self._extract_line_items_enhanced(combined_text, structured_data["line_items"])

        # Extract additional fields using fallback methods if needed
        self._extract_additional_fields_fallback(combined_text, structured_data)

        # Calculate overall extraction confidence
        overall_confidence = self._calculate_overall_extraction_confidence(structured_data)
        structured_data["extraction_metadata"]["overall_confidence"] = overall_confidence

        logger.info(f"Enhanced extraction completed with {overall_confidence:.1f}% confidence")
        return structured_data

    def _map_extracted_fields_to_structure(self, extracted_fields: Dict[str, Any], structured_data: Dict[str, Any]) -> None:
        """
        Map dynamically extracted fields to the structured data format.

        Args:
            extracted_fields: Fields extracted by dynamic pattern detector
            structured_data: Target structured data dictionary
        """
        # Map invoice header fields
        header_mapping = {
            'invoice_number': 'invoice_number',
            'invoice_date': 'invoice_date',
            'order_number': 'order_number',
            'customer_po': 'customer_po'
        }

        for field_key, struct_key in header_mapping.items():
            if field_key in extracted_fields:
                field_data = extracted_fields[field_key]
                structured_data["invoice_header"][struct_key] = field_data['value']
                structured_data["extraction_metadata"]["confidence_scores"][struct_key] = field_data['confidence']
                structured_data["extraction_metadata"]["patterns_used"][struct_key] = field_data['pattern']

        # Map vendor info
        if 'vendor_name' in extracted_fields:
            vendor_data = extracted_fields['vendor_name']
            structured_data["vendor_info"]["company_name"] = vendor_data['value']
            structured_data["extraction_metadata"]["confidence_scores"]["vendor_name"] = vendor_data['confidence']

        # Map customer info
        if 'customer_name' in extracted_fields:
            customer_data = extracted_fields['customer_name']
            structured_data["customer_info"]["company_name"] = customer_data['value']
            structured_data["extraction_metadata"]["confidence_scores"]["customer_name"] = customer_data['confidence']

        # Map totals
        totals_mapping = {
            'total_amount': 'invoice_total',
            'subtotal': 'subtotal',
            'tax_amount': 'tax'
        }

        for field_key, struct_key in totals_mapping.items():
            if field_key in extracted_fields:
                total_data = extracted_fields[field_key]
                structured_data["totals"][struct_key] = total_data['value']
                structured_data["extraction_metadata"]["confidence_scores"][struct_key] = total_data['confidence']

        # Map payment terms
        if 'payment_terms' in extracted_fields:
            terms_data = extracted_fields['payment_terms']
            structured_data["payment_terms"]["terms"] = terms_data['value']
            structured_data["extraction_metadata"]["confidence_scores"]["payment_terms"] = terms_data['confidence']

    def _extract_line_items_enhanced(self, text: str, line_items: List[Dict[str, Any]]) -> None:
        """
        Enhanced line item extraction using multiple strategies.

        Args:
            text: Input text to analyze
            line_items: List to populate with extracted line items
        """
        logger.info("Starting enhanced line item extraction")

        # Strategy 1: Use existing pattern-based extraction
        self._extract_line_items(text, line_items)

        # Strategy 2: Table structure detection (if no items found)
        if not line_items:
            self._extract_line_items_table_detection(text, line_items)

        # Strategy 3: Fallback to simple line-by-line analysis
        if not line_items:
            self._extract_line_items_simple_fallback(text, line_items)

        logger.info(f"Enhanced line item extraction found {len(line_items)} items")

    def _extract_line_items_table_detection(self, text: str, line_items: List[Dict[str, Any]]) -> None:
        """
        Extract line items using table structure detection.

        Args:
            text: Input text to analyze
            line_items: List to populate with extracted line items
        """
        lines = text.split('\n')

        # Look for table headers
        header_patterns = [
            r'(?:line|item|qty|quantity|description|price|amount|total)',
            r'(?:part|product|code|number)',
            r'(?:unit|each|extended)'
        ]

        table_start = -1
        for i, line in enumerate(lines):
            header_matches = sum(1 for pattern in header_patterns
                               if re.search(pattern, line, re.IGNORECASE))
            if header_matches >= 2:  # Found likely table header
                table_start = i
                break

        if table_start >= 0:
            # Process lines after header
            for i in range(table_start + 1, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue

                # Try to parse as table row
                item = self._parse_table_row(line, len(line_items) + 1)
                if item and self._validate_ocr_line_item(item):
                    line_items.append(item)

    def _parse_table_row(self, line: str, line_number: int) -> Optional[Dict[str, Any]]:
        """
        Parse a single table row into a line item.

        Args:
            line: Text line to parse
            line_number: Line number for the item

        Returns:
            Parsed line item or None if parsing failed
        """
        # Split by multiple spaces or tabs
        parts = re.split(r'\s{2,}|\t+', line.strip())

        if len(parts) < 3:
            return None

        # Try to identify numeric values
        numeric_parts = []
        text_parts = []

        for part in parts:
            if re.match(r'^[\d.,]+$', part.replace('$', '')):
                try:
                    numeric_parts.append(float(part.replace(',', '').replace('$', '')))
                except ValueError:
                    text_parts.append(part)
            else:
                text_parts.append(part)

        if len(numeric_parts) < 2:  # Need at least quantity and price
            return None

        # Build line item
        item = {
            "line_number": str(line_number),
            "description": " ".join(text_parts[:2]) if len(text_parts) >= 2 else text_parts[0] if text_parts else "",
            "item_code": text_parts[0] if text_parts else "",
            "quantity_ordered": int(numeric_parts[0]) if numeric_parts else 0,
            "quantity_shipped": int(numeric_parts[0]) if numeric_parts else 0,
            "unit_price": numeric_parts[-2] if len(numeric_parts) >= 2 else 0.0,
            "extended_amount": numeric_parts[-1] if numeric_parts else 0.0,
            "plant": "",
            "list_price": 0.0,
            "discount_percent": 0.0
        }

        return item

    def _extract_line_items_simple_fallback(self, text: str, line_items: List[Dict[str, Any]]) -> None:
        """
        Simple fallback method for line item extraction.

        Args:
            text: Input text to analyze
            line_items: List to populate with extracted line items
        """
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for lines with multiple numbers (likely line items)
            numbers = re.findall(r'\d+\.?\d*', line)
            if len(numbers) >= 3:  # At least qty, price, total
                item = {
                    "line_number": str(len(line_items) + 1),
                    "description": re.sub(r'\d+\.?\d*', '', line).strip(),
                    "item_code": "",
                    "quantity_ordered": int(float(numbers[0])) if numbers else 0,
                    "quantity_shipped": int(float(numbers[0])) if numbers else 0,
                    "unit_price": float(numbers[-2]) if len(numbers) >= 2 else 0.0,
                    "extended_amount": float(numbers[-1]) if numbers else 0.0,
                    "plant": "",
                    "list_price": 0.0,
                    "discount_percent": 0.0
                }

                if self._validate_ocr_line_item(item):
                    line_items.append(item)

    def _extract_additional_fields_fallback(self, text: str, structured_data: Dict[str, Any]) -> None:
        """
        Extract additional fields using fallback methods if dynamic patterns missed them.

        Args:
            text: Input text to analyze
            structured_data: Structured data dictionary to enhance
        """
        # Only use fallback if confidence is low or fields are missing
        metadata = structured_data.get("extraction_metadata", {})
        confidence_scores = metadata.get("confidence_scores", {})

        # Check for missing critical fields
        critical_fields = ["invoice_number", "invoice_date", "invoice_total"]
        missing_fields = [field for field in critical_fields
                         if field not in confidence_scores or confidence_scores[field] < 50]

        if missing_fields:
            logger.info(f"Using fallback extraction for missing fields: {missing_fields}")

            # Use original extraction methods as fallback
            if "invoice_number" in missing_fields or "invoice_date" in missing_fields:
                self._extract_invoice_header(text, structured_data["invoice_header"])

            if "invoice_total" in missing_fields:
                self._extract_totals(text, structured_data["totals"])

    def _calculate_overall_extraction_confidence(self, structured_data: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score for the extraction.

        Args:
            structured_data: Extracted structured data

        Returns:
            Overall confidence score (0-100)
        """
        metadata = structured_data.get("extraction_metadata", {})
        confidence_scores = metadata.get("confidence_scores", {})

        if not confidence_scores:
            return 0.0

        # Weight different field types
        field_weights = {
            "invoice_number": 0.2,
            "invoice_date": 0.15,
            "vendor_name": 0.1,
            "customer_name": 0.1,
            "invoice_total": 0.2,
            "subtotal": 0.1,
            "tax": 0.05,
            "payment_terms": 0.1
        }

        weighted_score = 0.0
        total_weight = 0.0

        for field, weight in field_weights.items():
            if field in confidence_scores:
                weighted_score += confidence_scores[field] * weight
                total_weight += weight

        # Add line items confidence
        line_items = structured_data.get("line_items", [])
        if line_items:
            line_items_confidence = min(100, len(line_items) * 20)  # 20 points per item, max 100
            weighted_score += line_items_confidence * 0.2
            total_weight += 0.2

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _extract_invoice_header(self, text: str, header_dict: Dict[str, Any]) -> None:
        """Extract invoice header information."""
        patterns = {
            "invoice_number": [
                r"invoice\s*number\s*:?\s*([0-9]{8})",
                r"invoice\s*#?\s*:?\s*([0-9]{8})",
                r"\b([0-9]{8})\b"
            ],
            "invoice_date": [
                r"invoice\s*date\s*:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
                r"date\s*:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
                r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b"
            ],
            "order_number": [
                r"order\s*no\s*:?\s*([A-Z0-9\-]+)",
                r"order\s*number\s*:?\s*([A-Z0-9\-]+)"
            ],
            "customer_po": [
                r"customer\s*po\s*:?\s*([A-Z0-9\-]+)",
                r"po\s*#?\s*:?\s*([A-Z0-9\-]+)"
            ]
        }

        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    header_dict[field] = match.group(1)
                    break

    def _extract_vendor_info(self, text: str, vendor_dict: Dict[str, Any]) -> None:
        """Extract vendor information."""
        patterns = {
            "company_name": r"(Ceco\s+Door[^\\n]*)",
            "address": r"(\d+\s+[A-Z\s]+DR\.?)",
            "city_state_zip": r"([A-Z]+,\s+[A-Z]{2}\s+\d{5})",
            "phone": r"(\(\d{3}\)\s+\d{3}-\d{4})"
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                vendor_dict[field] = match.group(1).strip()

    def _extract_customer_info(self, text: str, customer_dict: Dict[str, Any]) -> None:
        """Extract customer information."""
        patterns = {
            "sold_to_id": r"sold\s+to\s*:?\s*(\d+)",
            "ship_to_id": r"ship\s+to\s*:?\s*(\d+)",
            "company_name": r"(COOK\s+&?\s*BOARDMAN)",
            "address": r"(\d+\s+IMESON\s+PARK\s+BLVD)",
            "suite": r"(STE\s+\d+)",
            "city_state_zip": r"(JACKSONVILLE\s+FL\s+\d{5})"
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                customer_dict[field] = match.group(1).strip()

    def _extract_line_items(self, text: str, line_items: List[Dict[str, Any]]) -> None:
        """Extract line items from OCR text using multiple patterns."""

        # Multiple patterns for different invoice formats (same as text processor)
        patterns = [
            # CECO specific pattern that works best
            r"(\d{3})\s+(\d{3})\s+(FR3PC)\s+(\d+)\s+(\d+)\s+3\s+PIECE\s+FRAME\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",

            # Generic CECO format with FR3PC
            r"(\d{3})\s+(\d{3})\s+(FR3PC)\s+(\d+)\s+(\d+)\s+([^0-9]+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",

            # Generic line item pattern
            r"(\d{1,3})\s+(\d{2,3})\s+([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+([^0-9]+?)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)",

            # Alternative format with different spacing
            r"^(\d{1,3})\s+(\d{2,3})\s+([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+(.+?)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)$",

            # Simplified pattern for basic line items
            r"(\d{1,3})\s+([A-Z0-9]+)\s+(\d+)\s+([^0-9]+?)\s+([\d.,]+)\s+([\d.,]+)"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)

            for match in matches:
                line_item = self._parse_ocr_line_item_match(match)
                if line_item and self._validate_ocr_line_item(line_item):
                    # Check for duplicates
                    if not self._is_duplicate_ocr_line_item(line_item, line_items):
                        line_items.append(line_item)

            # If we found line items with this pattern, don't try others
            if line_items:
                break

        # If no structured patterns worked, try fallback extraction
        if not line_items:
            self._extract_ocr_line_items_fallback(text, line_items)

    def _parse_ocr_line_item_match(self, match: tuple) -> Dict[str, Any]:
        """Parse a regex match into a line item dictionary for OCR."""
        line_item = {}

        try:
            if len(match) >= 10:  # Full format with all fields
                line_item = {
                    "line_number": match[0].strip(),
                    "plant": match[1].strip(),
                    "item_code": match[2].strip(),
                    "quantity_ordered": self._safe_int_convert(match[3]),
                    "quantity_shipped": self._safe_int_convert(match[4]),
                    "description": match[5].strip(),
                    "list_price": self._safe_float_convert(match[6]),
                    "discount_percent": self._safe_float_convert(match[7]),
                    "unit_price": self._safe_float_convert(match[8]),
                    "extended_amount": self._safe_float_convert(match[9])
                }
            elif len(match) >= 6:  # Simplified format
                line_item = {
                    "line_number": match[0].strip(),
                    "item_code": match[1].strip(),
                    "quantity_ordered": self._safe_int_convert(match[2]),
                    "quantity_shipped": self._safe_int_convert(match[2]),
                    "description": match[3].strip(),
                    "unit_price": self._safe_float_convert(match[4]),
                    "extended_amount": self._safe_float_convert(match[5]),
                    "plant": "",
                    "list_price": 0.0,
                    "discount_percent": 0.0
                }
        except (IndexError, ValueError) as e:
            logger.warning(f"Error parsing OCR line item match: {e}")
            return {}

        return line_item

    def _safe_int_convert(self, value: str) -> int:
        """Safely convert string to integer."""
        try:
            return int(str(value).replace(',', '').strip())
        except (ValueError, AttributeError):
            return 0

    def _safe_float_convert(self, value: str) -> float:
        """Safely convert string to float."""
        try:
            return float(str(value).replace(',', '').replace('$', '').strip())
        except (ValueError, AttributeError):
            return 0.0

    def _validate_ocr_line_item(self, line_item: Dict[str, Any]) -> bool:
        """Validate that an OCR line item has required fields."""
        required_fields = ['line_number', 'description']
        numeric_fields = ['quantity_ordered', 'unit_price', 'extended_amount']

        # Check required fields are not empty
        for field in required_fields:
            if not line_item.get(field) or str(line_item[field]).strip() == "":
                return False

        # Check at least one numeric field has a valid value
        has_valid_numeric = any(
            isinstance(line_item.get(field), (int, float)) and line_item.get(field, 0) > 0
            for field in numeric_fields
        )

        return has_valid_numeric

    def _is_duplicate_ocr_line_item(self, new_item: Dict[str, Any], existing_items: List[Dict[str, Any]]) -> bool:
        """Check if OCR line item is a duplicate."""
        for existing in existing_items:
            if (existing.get('line_number') == new_item.get('line_number') and
                existing.get('item_code') == new_item.get('item_code')):
                return True
        return False

    def _extract_ocr_line_items_fallback(self, text: str, line_items: List[Dict[str, Any]]) -> None:
        """Fallback method to extract line items from OCR text."""
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for lines that might be line items
            if self._looks_like_ocr_line_item(line):
                line_item = self._parse_ocr_line_item_from_line(line, i + 1)
                if line_item and self._validate_ocr_line_item(line_item):
                    line_items.append(line_item)

    def _looks_like_ocr_line_item(self, line: str) -> bool:
        """Check if a line looks like it contains line item data."""
        patterns = [
            r'^\d{1,3}\s+\d{2,3}\s+[A-Z0-9]+',  # Line number, plant, item code
            r'^\d{1,3}\s+[A-Z0-9]+\s+\d+',      # Line number, item code, quantity
            r'\d+\s+\d+\s+[\d.,]+\s+[\d.,]+$',   # Quantities and prices at end
        ]

        return any(re.search(pattern, line) for pattern in patterns)

    def _parse_ocr_line_item_from_line(self, line: str, line_number: int) -> Dict[str, Any]:
        """Parse a single OCR line into a line item."""
        parts = line.split()

        if len(parts) < 4:
            return {}

        # Try to identify numeric values (prices, quantities)
        numeric_parts = []
        text_parts = []

        for part in parts:
            if re.match(r'^[\d.,]+$', part):
                numeric_parts.append(self._safe_float_convert(part))
            else:
                text_parts.append(part)

        if len(numeric_parts) < 2:  # Need at least quantity and price
            return {}

        return {
            "line_number": str(line_number),
            "item_code": text_parts[0] if text_parts else "",
            "description": " ".join(text_parts[1:]) if len(text_parts) > 1 else "",
            "quantity_ordered": int(numeric_parts[0]) if numeric_parts else 0,
            "quantity_shipped": int(numeric_parts[0]) if numeric_parts else 0,
            "unit_price": numeric_parts[-2] if len(numeric_parts) >= 2 else 0.0,
            "extended_amount": numeric_parts[-1] if numeric_parts else 0.0,
            "plant": "",
            "list_price": 0.0,
            "discount_percent": 0.0
        }

    def _extract_totals(self, text: str, totals_dict: Dict[str, Any]) -> None:
        """Extract totals information."""
        patterns = {
            "discount_amount": r"discount\s*of\s*\$\s*([\d,]+\.?\d*)",
            "total_sale": r"total\s*sale\s*:?\s*\$?\s*([\d,]+\.?\d*)",
            "tax": r"tax\s*:?\s*\$?\s*([\d,]+\.?\d*)",
            "invoice_total": r"invoice\s*total\s*\(USD\)\s*:?\s*\$?\s*([\d,]+\.?\d*)"
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                totals_dict[field] = match.group(1)

    def _extract_payment_terms(self, text: str, terms_dict: Dict[str, Any]) -> None:
        """Extract payment terms."""
        patterns = {
            "terms": r"(\d+%\s+\d+\s+DAYS,\s+NET\s+\d+)",
            "due_date": r"payable\s*on\s*(\d{1,2}\/\d{1,2}\/\d{2,4})",
            "discount_date": r"received\s*on\s*or\s*before\s*(\d{1,2}\/\d{1,2}\/\d{2,4})"
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                terms_dict[field] = match.group(1).strip()

    def _extract_shipping_info(self, text: str, shipping_dict: Dict[str, Any]) -> None:
        """Extract shipping information including freight charges."""
        patterns = {
            "tracking_number": [
                r"shipment\s*tracking\s*numb[er]*\s*:?\s*([A-Z0-9]+)",
                r"tracking\s*numb[er]*\s*:?\s*([A-Z0-9]+)"
            ],
            "carrier": [
                r"carrier\s*:?\s*(CUSTOMER\s+PICKUP[^\\n]*)",
                r"([A-Z\s]+LOGISTICS)(?:\s|$)",  # MAX TRANS LOGISTICS (stop at word boundary)
                r"([A-Z\s]+(?:LOGISTICS|TRANSPORT|SHIPPING|EXPRESS))(?:\s|$)",
                r"carrier\s*:?\s*([A-Z\s&]+(?:LOGISTICS|TRANSPORT|SHIPPING|EXPRESS))"
            ],
            "ship_from": [
                r"order\s*shipped\s*from\s*(\d+\s*-\s*[A-Z\s]+(?:MANUFACTURING|PLANT|FACILITY)?)",
                r"shipped\s*from\s*(\d+\s*-\s*[^\\n]+)"
            ],
            "shipping_method": [
                r"(PREPAID\s+3RD\s+PARTY)",
                r"(F\.O\.B\.\s+SHIP\s+POINT)",
                r"(CUSTOMER\s+PICKUP)",
                r"shipping\s*method\s*:?\s*([^\\n]+)"
            ],
            "freight_charge": [
                r"freight\s*charge\s*:?\s*\$?\s*([\d,]+\.?\d*)",
                r"freight\s*:?\s*\$?\s*([\d,]+\.?\d*)",
                r"shipping\s*charge\s*:?\s*\$?\s*([\d,]+\.?\d*)"
            ]
        }

        for field, pattern_list in patterns.items():
            if isinstance(pattern_list, str):
                pattern_list = [pattern_list]

            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    if field == "freight_charge":
                        # Clean and validate the amount
                        if value and re.match(r'^[\d,]+\.?\d*$', value):
                            shipping_dict[field] = value
                    else:
                        shipping_dict[field] = value
                    break
