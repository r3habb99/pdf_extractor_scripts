#!/usr/bin/env python3
"""
Vendor Detection Module

Automatically detects vendor type (CECO, SteelCraft, Schlage) from PDF content
and determines the appropriate processing method and output folder.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class VendorType(Enum):
    """Supported vendor types."""
    CECO = "ceco"
    STEELCRAFT = "steelcraft"
    SCHLAGE = "schlage"
    UNKNOWN = "unknown"


class VendorDetector:
    """
    Detects vendor type from PDF content and filename patterns.
    """
    
    def __init__(self):
        """Initialize the vendor detector with patterns."""
        self.vendor_patterns = {
            VendorType.CECO: {
                'filename_patterns': [
                    r'ceco',
                    r'F\d+[A-Z]*-I-\d+',  # CECO invoice pattern like F3970A-I-01792562
                ],
                'content_patterns': [
                    r'ceco\s+door\s+products',
                    r'ceco\s+door',
                    r'9159\s+BROOKVILLE\s+RD',  # CECO address
                    r'INDIANAPOLIS,?\s+IN\s+46239',
                    r'F\d+[A-Z]*-I-\d+',  # Invoice number pattern
                ],
                'confidence_keywords': [
                    'ceco', 'door products', 'brookville', 'indianapolis'
                ]
            },
            
            VendorType.STEELCRAFT: {
                'filename_patterns': [
                    r'steelcraft',
                    r'allegion.*steelcraft',
                ],
                'content_patterns': [
                    r'steelcraft',
                    r'allegion.*steelcraft',
                    r'steelcraft.*allegion',
                    r'9016\s+PRINCE\s+WILLIAM\s+ST',  # SteelCraft address
                    r'MANASSAS,?\s+VA\s+20110',
                ],
                'confidence_keywords': [
                    'steelcraft', 'allegion', 'manassas', 'prince william'
                ]
            },
            
            VendorType.SCHLAGE: {
                'filename_patterns': [
                    r'schlage',
                    r'\d+-\d+\s+schlage',
                ],
                'content_patterns': [
                    r'schlage\s+lock\s+co',
                    r'schlage.*allegion',
                    r'allegion.*schlage',
                    r'11819\s+NORTH\s+PENNSYLVANIA\s+STREET',  # Schlage address
                    r'CARMEL,?\s+IN\s+46032',
                    r'RES_CONTACT_CENTER@ALLEGION\.COM',
                ],
                'confidence_keywords': [
                    'schlage', 'allegion', 'carmel', 'pennsylvania street'
                ]
            }
        }
    
    def detect_vendor_from_filename(self, pdf_path: Path) -> Tuple[VendorType, float]:
        """
        Detect vendor type from filename.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (vendor_type, confidence_score)
        """
        filename = pdf_path.name.lower()
        
        for vendor_type, patterns in self.vendor_patterns.items():
            for pattern in patterns['filename_patterns']:
                if re.search(pattern, filename, re.IGNORECASE):
                    logger.info(f"Filename pattern match: {vendor_type.value} (pattern: {pattern})")
                    return vendor_type, 0.8  # High confidence from filename
        
        return VendorType.UNKNOWN, 0.0
    
    def detect_vendor_from_content(self, text_content: str) -> Tuple[VendorType, float]:
        """
        Detect vendor type from PDF text content.
        
        Args:
            text_content: Extracted text from PDF
            
        Returns:
            Tuple of (vendor_type, confidence_score)
        """
        if not text_content:
            return VendorType.UNKNOWN, 0.0
        
        text_lower = text_content.lower()
        vendor_scores = {}
        
        # Check each vendor's patterns
        for vendor_type, patterns in self.vendor_patterns.items():
            score = 0.0
            matches = 0
            
            # Check content patterns
            for pattern in patterns['content_patterns']:
                if re.search(pattern, text_content, re.IGNORECASE):
                    score += 0.3
                    matches += 1
                    logger.debug(f"Content pattern match for {vendor_type.value}: {pattern}")
            
            # Check confidence keywords
            for keyword in patterns['confidence_keywords']:
                if keyword.lower() in text_lower:
                    score += 0.1
                    matches += 1
                    logger.debug(f"Keyword match for {vendor_type.value}: {keyword}")
            
            if matches > 0:
                vendor_scores[vendor_type] = min(score, 1.0)  # Cap at 1.0
        
        if not vendor_scores:
            return VendorType.UNKNOWN, 0.0
        
        # Return vendor with highest score
        best_vendor = max(vendor_scores.items(), key=lambda x: x[1])
        logger.info(f"Content detection result: {best_vendor[0].value} (confidence: {best_vendor[1]:.2f})")
        
        return best_vendor[0], best_vendor[1]
    
    def detect_vendor(self, pdf_path: Path, text_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive vendor detection using both filename and content.
        
        Args:
            pdf_path: Path to the PDF file
            text_content: Optional extracted text content
            
        Returns:
            Detection result dictionary
        """
        logger.info(f"Detecting vendor for: {pdf_path.name}")
        
        # Try filename detection first
        filename_vendor, filename_confidence = self.detect_vendor_from_filename(pdf_path)
        
        # Try content detection if text is available
        content_vendor, content_confidence = VendorType.UNKNOWN, 0.0
        if text_content:
            content_vendor, content_confidence = self.detect_vendor_from_content(text_content)
        
        # Combine results
        final_vendor = VendorType.UNKNOWN
        final_confidence = 0.0
        detection_method = "unknown"
        
        # Prioritize content detection if confidence is high
        if content_confidence >= 0.6:
            final_vendor = content_vendor
            final_confidence = content_confidence
            detection_method = "content"
        elif filename_confidence >= 0.5:
            final_vendor = filename_vendor
            final_confidence = filename_confidence
            detection_method = "filename"
        elif content_confidence > 0:
            final_vendor = content_vendor
            final_confidence = content_confidence
            detection_method = "content_low_confidence"
        
        # If both methods agree, boost confidence
        if (filename_vendor == content_vendor and 
            filename_vendor != VendorType.UNKNOWN):
            final_confidence = min(filename_confidence + content_confidence, 1.0)
            detection_method = "combined"
        
        result = {
            'vendor_type': final_vendor,
            'vendor_name': final_vendor.value,
            'confidence': final_confidence,
            'detection_method': detection_method,
            'filename_detection': {
                'vendor': filename_vendor,
                'confidence': filename_confidence
            },
            'content_detection': {
                'vendor': content_vendor,
                'confidence': content_confidence
            },
            'output_folder': self._get_output_folder(final_vendor),
            'processor_type': self._get_processor_type(final_vendor)
        }
        
        logger.info(f"Final detection: {final_vendor.value} "
                   f"(confidence: {final_confidence:.2f}, method: {detection_method})")
        
        return result
    
    def _get_output_folder(self, vendor_type: VendorType) -> str:
        """Get the appropriate output folder for vendor type."""
        folder_mapping = {
            VendorType.CECO: "output/ceco",
            VendorType.STEELCRAFT: "output/steelcraft", 
            VendorType.SCHLAGE: "output/schlage",
            VendorType.UNKNOWN: "output/unknown"
        }
        return folder_mapping.get(vendor_type, "output/unknown")
    
    def _get_processor_type(self, vendor_type: VendorType) -> str:
        """Get the appropriate processor type for vendor."""
        processor_mapping = {
            VendorType.CECO: "ceco_specialized",
            VendorType.STEELCRAFT: "steelcraft_specialized",
            VendorType.SCHLAGE: "schlage_specialized",
            VendorType.UNKNOWN: "standard"
        }
        return processor_mapping.get(vendor_type, "standard")


def detect_vendor_type(pdf_path: Path, text_content: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for vendor detection.
    
    Args:
        pdf_path: Path to the PDF file
        text_content: Optional extracted text content
        
    Returns:
        Detection result dictionary
    """
    detector = VendorDetector()
    return detector.detect_vendor(pdf_path, text_content)


if __name__ == "__main__":
    # Test the vendor detector
    import sys
    
    if len(sys.argv) > 1:
        test_path = Path(sys.argv[1])
        if test_path.exists():
            result = detect_vendor_type(test_path)
            print(f"Vendor Detection Result:")
            print(f"  Vendor: {result['vendor_name']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Method: {result['detection_method']}")
            print(f"  Output Folder: {result['output_folder']}")
            print(f"  Processor: {result['processor_type']}")
        else:
            print(f"File not found: {test_path}")
    else:
        print("Usage: python vendor_detector.py <pdf_path>")
