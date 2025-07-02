"""
Enhanced PDF Text Detection Module

This module provides advanced functionality to detect whether a PDF contains selectable text
or is image-based (scanned document) and determines the best extraction method.
Enhanced with vendor-specific optimization, content quality assessment, and dynamic thresholds.
"""

import PyPDF2
import pdfplumber
import logging
import re
import statistics
from typing import Dict, Tuple, Optional, Any, List, Union
from pathlib import Path
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VendorType(Enum):
    """Supported vendor types for specialized processing."""
    CECO = "ceco"
    STEELCRAFT = "steelcraft"
    SCHLAGE = "schlage"
    UNKNOWN = "unknown"


class ContentQuality(Enum):
    """Content quality levels for extracted text."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CORRUPTED = "corrupted"


class PDFTextDetector:
    """
    Enhanced PDF text detector with vendor-specific optimization and content quality assessment.
    Provides intelligent methods to determine the best extraction approach.
    """

    def __init__(self,
                 vendor_type: Optional[VendorType] = None,
                 min_text_threshold: Optional[int] = None,
                 min_text_ratio: Optional[float] = None,
                 enable_quality_assessment: bool = True,
                 enable_page_analysis: bool = True):
        """
        Initialize the enhanced PDF text detector.

        Args:
            vendor_type: Specific vendor type for optimized processing
            min_text_threshold: Minimum number of characters (auto-set if None)
            min_text_ratio: Minimum ratio of text-containing pages (auto-set if None)
            enable_quality_assessment: Enable content quality validation
            enable_page_analysis: Enable detailed page-level analysis
        """
        self.vendor_type = vendor_type
        self.enable_quality_assessment = enable_quality_assessment
        self.enable_page_analysis = enable_page_analysis

        # Set vendor-specific thresholds
        thresholds = self._get_vendor_thresholds(vendor_type)
        self.min_text_threshold = min_text_threshold or thresholds['min_text_threshold']
        self.min_text_ratio = min_text_ratio or thresholds['min_text_ratio']
        self.quality_threshold = thresholds['quality_threshold']

        # Cache for detection results
        self._detection_cache = {}

        logger.info(f"Initialized PDFTextDetector for vendor: {vendor_type}, "
                   f"thresholds: {self.min_text_threshold}/{self.min_text_ratio}")

    def _get_vendor_thresholds(self, vendor_type: Optional[VendorType]) -> Dict[str, Any]:
        """
        Get vendor-specific detection thresholds optimized for each invoice type.

        Args:
            vendor_type: The vendor type to get thresholds for

        Returns:
            Dictionary with optimized thresholds
        """
        vendor_configs = {
            VendorType.CECO: {
                'min_text_threshold': 100,  # CECO invoices are text-rich
                'min_text_ratio': 0.8,     # Most pages should have text
                'quality_threshold': 0.7,   # High quality expected
                'description': 'CECO invoices typically have excellent text extraction'
            },
            VendorType.STEELCRAFT: {
                'min_text_threshold': 75,   # Mixed content possible
                'min_text_ratio': 0.6,     # Some pages may be image-heavy
                'quality_threshold': 0.6,   # Medium quality acceptable
                'description': 'Steelcraft invoices may have mixed content'
            },
            VendorType.SCHLAGE: {
                'min_text_threshold': 30,   # Often image-based
                'min_text_ratio': 0.3,     # Low text ratio expected
                'quality_threshold': 0.5,   # Lower quality threshold
                'description': 'Schlage invoices often require OCR processing'
            },
            VendorType.UNKNOWN: {
                'min_text_threshold': 50,   # Conservative default
                'min_text_ratio': 0.5,     # Balanced approach
                'quality_threshold': 0.6,   # Medium quality
                'description': 'Unknown vendor - using balanced thresholds'
            }
        }

        key = vendor_type if vendor_type is not None else VendorType.UNKNOWN
        return vendor_configs.get(key, vendor_configs[VendorType.UNKNOWN])

    def detect_pdf_type(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Enhanced PDF analysis with caching, quality assessment, and vendor optimization.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing enhanced detection results and recommendations
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Check cache first
        cache_key = f"{pdf_path}_{pdf_path.stat().st_mtime}"
        if cache_key in self._detection_cache:
            logger.info(f"Using cached detection result for {pdf_path.name}")
            return self._detection_cache[cache_key]

        logger.info(f"Analyzing PDF: {pdf_path.name} (vendor: {self.vendor_type})")

        # Enhanced detection with multiple methods
        pypdf2_result = self._detect_with_pypdf2(pdf_path)
        pdfplumber_result = self._detect_with_pdfplumber(pdf_path)

        # Page-level analysis if enabled
        page_analysis = {}
        if self.enable_page_analysis:
            page_analysis = self._analyze_page_content(pdf_path, pdfplumber_result)

        # Combine results with enhanced logic
        final_result = self._combine_detection_results_enhanced(
            pypdf2_result, pdfplumber_result, page_analysis, pdf_path
        )

        # Cache the result
        self._detection_cache[cache_key] = final_result

        logger.info(f"Detection complete for {pdf_path.name}: {final_result['pdf_type']} "
                   f"(confidence: {final_result['confidence']}, "
                   f"quality: {final_result.get('content_quality', 'unknown')})")
        return final_result
    
    def _detect_with_pypdf2(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Detect text using PyPDF2.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PyPDF2 detection results
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                total_text = ""
                pages_with_text = 0
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text().strip()
                        total_text += page_text
                        
                        if len(page_text) > self.min_text_threshold:
                            pages_with_text += 1
                            
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1} with PyPDF2: {e}")
                        continue
                
                text_ratio = pages_with_text / total_pages if total_pages > 0 else 0
                
                return {
                    'method': 'PyPDF2',
                    'total_pages': total_pages,
                    'pages_with_text': pages_with_text,
                    'total_characters': len(total_text),
                    'text_ratio': text_ratio,
                    'has_selectable_text': (
                        len(total_text) > self.min_text_threshold and 
                        text_ratio >= self.min_text_ratio
                    ),
                    'sample_text': total_text[:200] if total_text else "",
                    'success': True
                }
                
        except Exception as e:
            logger.error(f"PyPDF2 detection failed: {e}")
            return {
                'method': 'PyPDF2',
                'success': False,
                'error': str(e),
                'has_selectable_text': False
            }

    def _validate_text_quality(self, text: str) -> Tuple[ContentQuality, float, Dict[str, Any]]:
        """
        Assess the quality of extracted text to determine if it's meaningful or corrupted.

        Args:
            text: The extracted text to analyze

        Returns:
            Tuple of (quality_level, quality_score, quality_metrics)
        """
        if not text or len(text.strip()) < 10:
            return ContentQuality.CORRUPTED, 0.0, {'reason': 'insufficient_text'}

        metrics = {}
        quality_indicators = []

        # 1. Character distribution analysis
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        digit_chars = sum(1 for c in text if c.isdigit())
        space_chars = sum(1 for c in text if c.isspace())
        special_chars = total_chars - alpha_chars - digit_chars - space_chars

        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        digit_ratio = digit_chars / total_chars if total_chars > 0 else 0
        special_ratio = special_chars / total_chars if total_chars > 0 else 0

        metrics.update({
            'alpha_ratio': alpha_ratio,
            'digit_ratio': digit_ratio,
            'special_ratio': special_ratio,
            'total_chars': total_chars
        })

        # Good text should have reasonable alpha ratio
        if 0.3 <= alpha_ratio <= 0.8:
            quality_indicators.append(0.8)
        elif 0.1 <= alpha_ratio < 0.3 or 0.8 < alpha_ratio <= 0.95:
            quality_indicators.append(0.5)
        else:
            quality_indicators.append(0.1)

        # 2. Word structure analysis
        words = text.split()
        if words:
            avg_word_length = statistics.mean(len(word) for word in words)
            valid_words = sum(1 for word in words if 2 <= len(word) <= 20 and word.isalnum())
            word_validity_ratio = valid_words / len(words)

            metrics.update({
                'word_count': len(words),
                'avg_word_length': avg_word_length,
                'word_validity_ratio': word_validity_ratio
            })

            # Good word structure indicators
            if 3 <= avg_word_length <= 8 and word_validity_ratio >= 0.6:
                quality_indicators.append(0.9)
            elif 2 <= avg_word_length <= 12 and word_validity_ratio >= 0.4:
                quality_indicators.append(0.6)
            else:
                quality_indicators.append(0.3)
        else:
            quality_indicators.append(0.1)
            metrics.update({'word_count': 0, 'avg_word_length': 0, 'word_validity_ratio': 0})

        # 3. Invoice-specific pattern detection
        invoice_patterns = [
            r'\b(?:invoice|bill|receipt)\b',
            r'\b(?:total|amount|price|cost)\b',
            r'\b(?:date|qty|quantity)\b',
            r'\$\d+\.?\d*',
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b'  # Currency amounts
        ]

        pattern_matches = 0
        for pattern in invoice_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_matches += 1

        pattern_score = min(pattern_matches / len(invoice_patterns), 1.0)
        quality_indicators.append(pattern_score)
        metrics['invoice_pattern_score'] = pattern_score

        # 4. Repetitive character detection (OCR artifacts)
        repetitive_chars = len(re.findall(r'(.)\1{4,}', text))  # 5+ repeated chars
        repetitive_ratio = repetitive_chars / total_chars if total_chars > 0 else 0

        if repetitive_ratio < 0.05:
            quality_indicators.append(0.8)
        elif repetitive_ratio < 0.15:
            quality_indicators.append(0.5)
        else:
            quality_indicators.append(0.2)

        metrics['repetitive_ratio'] = repetitive_ratio

        # Calculate overall quality score
        overall_score = statistics.mean(quality_indicators) if quality_indicators else 0.0

        # Determine quality level
        if overall_score >= 0.7:
            quality_level = ContentQuality.HIGH
        elif overall_score >= 0.5:
            quality_level = ContentQuality.MEDIUM
        elif overall_score >= 0.3:
            quality_level = ContentQuality.LOW
        else:
            quality_level = ContentQuality.CORRUPTED

        metrics['overall_score'] = overall_score
        metrics['quality_indicators'] = quality_indicators

        return quality_level, overall_score, metrics

    def _detect_with_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Detect text using pdfplumber (more accurate for complex layouts).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with pdfplumber detection results
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                total_text = ""
                pages_with_text = 0
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            page_text = page_text.strip()
                            total_text += page_text
                            
                            if len(page_text) > self.min_text_threshold:
                                pages_with_text += 1
                                
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1} with pdfplumber: {e}")
                        continue
                
                text_ratio = pages_with_text / total_pages if total_pages > 0 else 0
                
                return {
                    'method': 'pdfplumber',
                    'total_pages': total_pages,
                    'pages_with_text': pages_with_text,
                    'total_characters': len(total_text),
                    'text_ratio': text_ratio,
                    'has_selectable_text': (
                        len(total_text) > self.min_text_threshold and 
                        text_ratio >= self.min_text_ratio
                    ),
                    'sample_text': total_text[:200] if total_text else "",
                    'success': True
                }
                
        except Exception as e:
            logger.error(f"pdfplumber detection failed: {e}")
            return {
                'method': 'pdfplumber',
                'success': False,
                'error': str(e),
                'has_selectable_text': False
            }

    def _analyze_page_content(self, pdf_path: Path, pdfplumber_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform detailed page-level analysis for mixed content detection.

        Args:
            pdf_path: Path to the PDF file
            pdfplumber_result: Results from pdfplumber detection

        Returns:
            Dictionary with page-level analysis results
        """
        page_analysis = {
            'pages': [],
            'mixed_content_detected': False,
            'text_pages': 0,
            'image_pages': 0,
            'hybrid_pages': 0,
            'page_quality_scores': []
        }

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_info = {
                        'page_number': page_num + 1,
                        'has_text': False,
                        'text_length': 0,
                        'has_images': False,
                        'has_tables': False,
                        'content_quality': ContentQuality.CORRUPTED,
                        'quality_score': 0.0,
                        'recommended_method': 'ocr'
                    }

                    try:
                        # Extract text for this page
                        page_text = page.extract_text()
                        if page_text:
                            page_text = page_text.strip()
                            page_info['text_length'] = len(page_text)

                            if len(page_text) > self.min_text_threshold:
                                page_info['has_text'] = True

                                # Assess text quality if enabled
                                if self.enable_quality_assessment:
                                    quality_level, quality_score, _ = self._validate_text_quality(page_text)
                                    page_info['content_quality'] = quality_level
                                    page_info['quality_score'] = quality_score
                                    page_analysis['page_quality_scores'].append(quality_score)

                                    # Recommend method based on quality
                                    if quality_score >= self.quality_threshold:
                                        page_info['recommended_method'] = 'text_extraction'
                                    else:
                                        page_info['recommended_method'] = 'ocr_with_text_fallback'

                        # Check for images (basic detection)
                        if hasattr(page, 'images') and page.images:
                            page_info['has_images'] = True

                        # Check for tables
                        if hasattr(page, 'extract_tables'):
                            tables = page.extract_tables()
                            if tables:
                                page_info['has_tables'] = True

                        # Classify page type
                        if page_info['has_text'] and page_info['has_images']:
                            page_analysis['hybrid_pages'] += 1
                            page_analysis['mixed_content_detected'] = True
                        elif page_info['has_text']:
                            page_analysis['text_pages'] += 1
                        else:
                            page_analysis['image_pages'] += 1

                    except Exception as e:
                        logger.warning(f"Error analyzing page {page_num + 1}: {e}")
                        page_info['error'] = str(e)

                    page_analysis['pages'].append(page_info)

        except Exception as e:
            logger.error(f"Page analysis failed: {e}")
            page_analysis['error'] = str(e)

        return page_analysis
    
    def _combine_detection_results_enhanced(
        self,
        pypdf2_result: Dict[str, Any],
        pdfplumber_result: Dict[str, Any],
        page_analysis: Dict[str, Any],
        pdf_path: Path
    ) -> Dict[str, Any]:
        """
        Enhanced combination of detection results with quality assessment and vendor optimization.

        Args:
            pypdf2_result: Results from PyPDF2 detection
            pdfplumber_result: Results from pdfplumber detection
            page_analysis: Results from page-level analysis
            pdf_path: Path to the PDF file

        Returns:
            Enhanced detection result with quality metrics and optimized recommendations
        """
        # Determine if either method found selectable text
        pypdf2_has_text = pypdf2_result.get('has_selectable_text', False)
        pdfplumber_has_text = pdfplumber_result.get('has_selectable_text', False)

        # Prioritize pdfplumber results (more accurate for complex layouts)
        if pdfplumber_result.get('success', False):
            primary_result = pdfplumber_result
            has_selectable_text = pdfplumber_has_text
            primary_method = 'pdfplumber'
        elif pypdf2_result.get('success', False):
            primary_result = pypdf2_result
            has_selectable_text = pypdf2_has_text
            primary_method = 'pypdf2'
        else:
            # Both methods failed - assume image-based
            has_selectable_text = False
            primary_result = {'total_pages': 0, 'total_characters': 0, 'sample_text': ''}
            primary_method = 'none'

        # Content quality assessment
        content_quality = ContentQuality.CORRUPTED
        quality_score = 0.0
        quality_metrics = {}

        if self.enable_quality_assessment and primary_result.get('sample_text'):
            content_quality, quality_score, quality_metrics = self._validate_text_quality(
                primary_result['sample_text']
            )

        # Enhanced confidence calculation
        confidence_factors = []

        # Method agreement factor
        if pypdf2_has_text and pdfplumber_has_text:
            confidence_factors.append(0.9)
        elif pypdf2_has_text or pdfplumber_has_text:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.8)  # Both agree on no text

        # Quality factor
        if quality_score > 0:
            confidence_factors.append(min(quality_score + 0.2, 1.0))

        # Vendor-specific factor
        vendor_confidence = self._get_vendor_confidence_adjustment(
            primary_result, content_quality, quality_score
        )
        confidence_factors.append(vendor_confidence)

        # Page analysis factor
        if page_analysis and page_analysis.get('page_quality_scores'):
            avg_page_quality = statistics.mean(page_analysis['page_quality_scores'])
            confidence_factors.append(avg_page_quality)

        overall_confidence = statistics.mean(confidence_factors) if confidence_factors else 0.5

        # Determine PDF type and processing method
        pdf_type, recommended_method = self._determine_processing_strategy(
            has_selectable_text, content_quality, quality_score, page_analysis, overall_confidence
        )

        # Enhanced mixed content detection
        mixed_content = self._detect_mixed_content(primary_result, page_analysis)

        return {
            'pdf_path': str(pdf_path),
            'pdf_type': pdf_type,
            'recommended_method': recommended_method,
            'confidence': self._categorize_confidence(overall_confidence),
            'confidence_score': overall_confidence,
            'mixed_content': mixed_content,
            'content_quality': content_quality.value,
            'quality_score': quality_score,
            'vendor_type': self.vendor_type.value if self.vendor_type else 'unknown',
            'detection_details': {
                'pypdf2': pypdf2_result,
                'pdfplumber': pdfplumber_result,
                'primary_method': primary_method,
                'total_pages': primary_result.get('total_pages', 0),
                'total_characters': primary_result.get('total_characters', 0),
                'text_ratio': primary_result.get('text_ratio', 0),
                'sample_text': primary_result.get('sample_text', ""),
                'quality_metrics': quality_metrics,
                'page_analysis': page_analysis,
                'confidence_factors': confidence_factors
            }
        }

    def _get_vendor_confidence_adjustment(
        self,
        primary_result: Dict[str, Any],
        content_quality: ContentQuality,
        quality_score: float
    ) -> float:
        """
        Calculate vendor-specific confidence adjustment based on known characteristics.

        Args:
            primary_result: Primary detection result
            content_quality: Assessed content quality
            quality_score: Numeric quality score

        Returns:
            Confidence adjustment factor (0.0 to 1.0)
        """
        if not self.vendor_type:
            return 0.7  # Neutral confidence for unknown vendors

        text_ratio = primary_result.get('text_ratio', 0)
        total_chars = primary_result.get('total_characters', 0)

        if self.vendor_type == VendorType.CECO:
            # CECO invoices typically have excellent text extraction
            if text_ratio >= 0.8 and total_chars >= 1000:
                return 0.95
            elif text_ratio >= 0.6 and total_chars >= 500:
                return 0.85
            else:
                return 0.7  # Unexpected for CECO

        elif self.vendor_type == VendorType.STEELCRAFT:
            # Steelcraft may have mixed content
            if text_ratio >= 0.6 and content_quality in [ContentQuality.HIGH, ContentQuality.MEDIUM]:
                return 0.85
            elif text_ratio >= 0.3:
                return 0.75
            else:
                return 0.8  # OCR expected

        elif self.vendor_type == VendorType.SCHLAGE:
            # Schlage often requires OCR
            if text_ratio <= 0.4:
                return 0.9  # Expected behavior
            elif text_ratio >= 0.7 and content_quality == ContentQuality.HIGH:
                return 0.85  # Surprisingly good text
            else:
                return 0.75

        return 0.7  # Default

    def _determine_processing_strategy(
        self,
        has_selectable_text: bool,
        content_quality: ContentQuality,
        quality_score: float,
        page_analysis: Dict[str, Any],
        overall_confidence: float
    ) -> Tuple[str, str]:
        """
        Determine optimal processing strategy based on comprehensive analysis.

        Args:
            has_selectable_text: Whether text was detected
            content_quality: Quality of extracted text
            quality_score: Numeric quality score
            page_analysis: Page-level analysis results
            overall_confidence: Overall confidence score

        Returns:
            Tuple of (pdf_type, recommended_method)
        """
        # Check for mixed content from page analysis
        mixed_content = page_analysis.get('mixed_content_detected', False)
        text_pages = page_analysis.get('text_pages', 0)
        image_pages = page_analysis.get('image_pages', 0)
        total_pages = text_pages + image_pages + page_analysis.get('hybrid_pages', 0)

        if has_selectable_text and quality_score >= self.quality_threshold:
            pdf_type = "text_selectable"

            if mixed_content and image_pages > text_pages:
                recommended_method = "text_with_ocr_fallback"
            elif content_quality == ContentQuality.HIGH:
                recommended_method = "text_extraction"
            elif content_quality == ContentQuality.MEDIUM:
                recommended_method = "text_extraction"
            else:
                recommended_method = "text_with_ocr_fallback"

        elif has_selectable_text and quality_score < self.quality_threshold:
            pdf_type = "mixed_content"
            recommended_method = "ocr_with_text_fallback"

        else:
            pdf_type = "image_based"
            recommended_method = "ocr"

        # Vendor-specific strategy adjustments
        if self.vendor_type == VendorType.SCHLAGE and pdf_type == "text_selectable":
            # Schlage often benefits from OCR even with text
            recommended_method = "text_with_ocr_fallback"
        elif self.vendor_type == VendorType.CECO and pdf_type == "image_based":
            # CECO rarely needs OCR, might be detection error
            recommended_method = "text_with_ocr_fallback"

        return pdf_type, recommended_method

    def _detect_mixed_content(
        self,
        primary_result: Dict[str, Any],
        page_analysis: Dict[str, Any]
    ) -> bool:
        """
        Enhanced mixed content detection using multiple indicators.

        Args:
            primary_result: Primary detection result
            page_analysis: Page-level analysis results

        Returns:
            True if mixed content is detected
        """
        # Page analysis mixed content
        if page_analysis.get('mixed_content_detected', False):
            return True

        # Text ratio based detection
        text_ratio = primary_result.get('text_ratio', 0)
        if 0.1 < text_ratio < 0.9:
            return True

        # Page type distribution
        text_pages = page_analysis.get('text_pages', 0)
        image_pages = page_analysis.get('image_pages', 0)
        hybrid_pages = page_analysis.get('hybrid_pages', 0)

        if text_pages > 0 and (image_pages > 0 or hybrid_pages > 0):
            return True

        return False

    def _categorize_confidence(self, confidence_score: float) -> str:
        """
        Categorize numeric confidence score into descriptive levels.

        Args:
            confidence_score: Numeric confidence (0.0 to 1.0)

        Returns:
            Confidence category string
        """
        if confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.6:
            return "medium"
        elif confidence_score >= 0.4:
            return "low"
        else:
            return "very_low"

    def _combine_detection_results(
        self,
        pypdf2_result: Dict[str, Any],
        pdfplumber_result: Dict[str, Any],
        pdf_path: Path
    ) -> Dict[str, Any]:
        """
        Backward compatibility method for the old interface.
        Delegates to the enhanced method with empty page analysis.

        Args:
            pypdf2_result: Results from PyPDF2 detection
            pdfplumber_result: Results from pdfplumber detection
            pdf_path: Path to the PDF file

        Returns:
            Detection result using enhanced logic
        """
        return self._combine_detection_results_enhanced(
            pypdf2_result, pdfplumber_result, {}, pdf_path
        )


def detect_pdf_type(pdf_path: str, vendor_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Enhanced convenience function to detect PDF type with vendor optimization.

    Args:
        pdf_path: Path to the PDF file
        vendor_type: Optional vendor type string ('ceco', 'steelcraft', 'schlage')
        **kwargs: Additional arguments for PDFTextDetector

    Returns:
        Enhanced detection results dictionary
    """
    # Convert string vendor type to enum
    vendor_enum = None
    if vendor_type:
        try:
            vendor_enum = VendorType(vendor_type.lower())
        except ValueError:
            logger.warning(f"Unknown vendor type: {vendor_type}, using default settings")

    detector = PDFTextDetector(vendor_type=vendor_enum, **kwargs)
    return detector.detect_pdf_type(pdf_path)


if __name__ == "__main__":
    # Enhanced example usage with vendor support
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced PDF Text Detection with Vendor Optimization')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--vendor', choices=['ceco', 'steelcraft', 'schlage'],
                       help='Vendor type for optimized processing')
    parser.add_argument('--quality-assessment', action='store_true', default=True,
                       help='Enable content quality assessment')
    parser.add_argument('--page-analysis', action='store_true', default=True,
                       help='Enable detailed page-level analysis')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        result = detect_pdf_type(
            args.pdf_path,
            vendor_type=args.vendor,
            enable_quality_assessment=args.quality_assessment,
            enable_page_analysis=args.page_analysis
        )

        print(f"📄 PDF Analysis Results for: {args.pdf_path}")
        print(f"🔍 PDF Type: {result['pdf_type']}")
        print(f"⚙️  Recommended Method: {result['recommended_method']}")
        print(f"📊 Confidence: {result['confidence']} ({result.get('confidence_score', 0):.2f})")
        print(f"🏷️  Vendor: {result.get('vendor_type', 'unknown')}")
        print(f"✨ Content Quality: {result.get('content_quality', 'unknown')} ({result.get('quality_score', 0):.2f})")

        if result['mixed_content']:
            print("⚠️  Note: PDF contains mixed content (text and image pages)")

        if args.verbose:
            details = result.get('detection_details', {})
            print(f"\n📋 Detailed Analysis:")
            print(f"   Total Pages: {details.get('total_pages', 0)}")
            print(f"   Total Characters: {details.get('total_characters', 0)}")
            print(f"   Text Ratio: {details.get('text_ratio', 0):.2f}")

            page_analysis = details.get('page_analysis', {})
            if page_analysis:
                print(f"   Text Pages: {page_analysis.get('text_pages', 0)}")
                print(f"   Image Pages: {page_analysis.get('image_pages', 0)}")
                print(f"   Hybrid Pages: {page_analysis.get('hybrid_pages', 0)}")

    except Exception as e:
        print(f"❌ Error analyzing PDF: {e}")
        sys.exit(1)
