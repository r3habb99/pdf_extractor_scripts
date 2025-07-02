#!/usr/bin/env python3
"""
Enhanced Dynamic Pattern Detection for Invoice Processing

This module provides adaptive pattern recognition that works across different
invoice formats without hardcoded vendor-specific patterns.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FieldType(Enum):
    """Types of fields that can be extracted from invoices."""
    INVOICE_NUMBER = "invoice_number"
    INVOICE_DATE = "invoice_date"
    ORDER_NUMBER = "order_number"
    CUSTOMER_PO = "customer_po"
    VENDOR_NAME = "vendor_name"
    CUSTOMER_NAME = "customer_name"
    LINE_ITEM = "line_item"
    TOTAL_AMOUNT = "total_amount"
    SUBTOTAL = "subtotal"
    TAX_AMOUNT = "tax_amount"
    PAYMENT_TERMS = "payment_terms"


@dataclass
class PatternMatch:
    """Represents a pattern match with confidence score."""
    field_type: FieldType
    value: str
    confidence: float
    pattern_used: str
    context: str


class DynamicPatternDetector:
    """
    Advanced pattern detector that adapts to different invoice formats
    without hardcoded vendor-specific patterns.
    """
    
    def __init__(self):
        """Initialize the dynamic pattern detector."""
        self.field_patterns = self._initialize_adaptive_patterns()
        self.context_keywords = self._initialize_context_keywords()
    
    def _initialize_adaptive_patterns(self) -> Dict[FieldType, List[str]]:
        """
        Initialize adaptive patterns that work across different invoice formats.
        
        Returns:
            Dictionary mapping field types to pattern lists
        """
        return {
            FieldType.INVOICE_NUMBER: [
                # Generic invoice number patterns
                r'invoice\s*(?:number|no|#)?\s*:?\s*([A-Z0-9\-]{6,})',
                r'inv\s*(?:number|no|#)?\s*:?\s*([A-Z0-9\-]{6,})',
                r'(?:^|\s)([A-Z0-9\-]{8,12})(?=\s|$)',  # Standalone alphanumeric
                r'(?:document|doc)\s*(?:number|no|#)?\s*:?\s*([A-Z0-9\-]{6,})',
                r'(?:bill|billing)\s*(?:number|no|#)?\s*:?\s*([A-Z0-9\-]{6,})',
            ],
            
            FieldType.INVOICE_DATE: [
                # Various date formats
                r'invoice\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
                r'date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
                r'(?:bill|billing)\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
                r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',  # Standalone date
                r'(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})',  # ISO format
            ],
            
            FieldType.ORDER_NUMBER: [
                r'(?:order|purchase)\s*(?:number|no|#)?\s*:?\s*([A-Z0-9\-]{4,})',
                r'po\s*(?:number|no|#)?\s*:?\s*([A-Z0-9\-]{4,})',
                r'customer\s*po\s*:?\s*([A-Z0-9\-]{4,})',
                r'reference\s*(?:number|no|#)?\s*:?\s*([A-Z0-9\-]{4,})',
            ],
            
            FieldType.VENDOR_NAME: [
                # Company name patterns (adaptive)
                r'(?:from|vendor|supplier)\s*:?\s*([A-Z][A-Za-z\s&\.,]{10,50})',
                r'^([A-Z][A-Za-z\s&\.,]{10,50})(?:\s*(?:inc|corp|llc|ltd)\.?)?',
                r'bill\s*from\s*:?\s*([A-Z][A-Za-z\s&\.,]{10,50})',
            ],
            
            FieldType.CUSTOMER_NAME: [
                r'(?:to|customer|client|bill\s*to)\s*:?\s*([A-Z][A-Za-z\s&\.,]{5,50})',
                r'ship\s*to\s*:?\s*([A-Z][A-Za-z\s&\.,]{5,50})',
                r'sold\s*to\s*:?\s*([A-Z][A-Za-z\s&\.,]{5,50})',
            ],
            
            FieldType.TOTAL_AMOUNT: [
                r'(?:total|grand\s*total|invoice\s*total)\s*(?:\(USD\))?\s*:?\s*\$?\s*([\d,]+\.?\d*)',
                r'amount\s*due\s*:?\s*\$?\s*([\d,]+\.?\d*)',
                r'balance\s*due\s*:?\s*\$?\s*([\d,]+\.?\d*)',
                r'total\s*amount\s*:?\s*\$?\s*([\d,]+\.?\d*)',
            ],
            
            FieldType.SUBTOTAL: [
                r'(?:sub\s*total|subtotal)\s*:?\s*\$?\s*([\d,]+\.?\d*)',
                r'(?:net|before\s*tax)\s*:?\s*\$?\s*([\d,]+\.?\d*)',
                r'merchandise\s*total\s*:?\s*\$?\s*([\d,]+\.?\d*)',
            ],
            
            FieldType.TAX_AMOUNT: [
                r'(?:tax|sales\s*tax|vat)\s*:?\s*\$?\s*([\d,]+\.?\d*)',
                r'(?:state|local)\s*tax\s*:?\s*\$?\s*([\d,]+\.?\d*)',
            ],
            
            FieldType.PAYMENT_TERMS: [
                r'(?:terms|payment\s*terms)\s*:?\s*([^\\n]{10,50})',
                r'(?:net|due)\s*(\d+)\s*(?:days?)',
                r'(\d+%\s*\d+\s*days?,?\s*net\s*\d+)',
            ],
        }
    
    def _initialize_context_keywords(self) -> Dict[FieldType, List[str]]:
        """
        Initialize context keywords that help identify field types.
        
        Returns:
            Dictionary mapping field types to context keyword lists
        """
        return {
            FieldType.INVOICE_NUMBER: ['invoice', 'bill', 'document', 'number'],
            FieldType.INVOICE_DATE: ['date', 'issued', 'created', 'bill'],
            FieldType.ORDER_NUMBER: ['order', 'purchase', 'po', 'reference'],
            FieldType.VENDOR_NAME: ['from', 'vendor', 'supplier', 'company'],
            FieldType.CUSTOMER_NAME: ['to', 'customer', 'client', 'ship', 'bill'],
            FieldType.TOTAL_AMOUNT: ['total', 'amount', 'due', 'balance', 'grand'],
            FieldType.SUBTOTAL: ['subtotal', 'sub', 'net', 'before'],
            FieldType.TAX_AMOUNT: ['tax', 'vat', 'sales', 'state'],
            FieldType.PAYMENT_TERMS: ['terms', 'payment', 'net', 'due'],
        }
    
    def extract_all_fields(self, text: str) -> Dict[str, Any]:
        """
        Extract all possible fields from text using dynamic patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of extracted fields with confidence scores
        """
        extracted_fields = {}
        
        for field_type in FieldType:
            matches = self.extract_field(text, field_type)
            if matches:
                # Use the highest confidence match
                best_match = max(matches, key=lambda x: x.confidence)
                extracted_fields[field_type.value] = {
                    'value': best_match.value,
                    'confidence': best_match.confidence,
                    'pattern': best_match.pattern_used,
                    'context': best_match.context
                }
        
        return extracted_fields
    
    def extract_field(self, text: str, field_type: FieldType) -> List[PatternMatch]:
        """
        Extract a specific field type from text.
        
        Args:
            text: Input text to analyze
            field_type: Type of field to extract
            
        Returns:
            List of pattern matches for the field type
        """
        matches = []
        patterns = self.field_patterns.get(field_type, [])
        
        for pattern in patterns:
            try:
                regex_matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in regex_matches:
                    value = match.group(1).strip() if match.groups() else match.group(0).strip()
                    
                    if self._is_valid_field_value(field_type, value):
                        confidence = self._calculate_field_confidence(
                            field_type, value, pattern, text, match.start(), match.end()
                        )
                        
                        context = self._extract_context(text, match.start(), match.end())
                        
                        matches.append(PatternMatch(
                            field_type=field_type,
                            value=value,
                            confidence=confidence,
                            pattern_used=pattern,
                            context=context
                        ))
                        
            except re.error as e:
                logger.warning(f"Invalid regex pattern for {field_type}: {pattern} - {e}")
                continue
        
        # Remove duplicates and sort by confidence
        unique_matches = self._remove_duplicate_matches(matches)
        return sorted(unique_matches, key=lambda x: x.confidence, reverse=True)
    
    def _is_valid_field_value(self, field_type: FieldType, value: str) -> bool:
        """
        Validate if a value is reasonable for the given field type.
        
        Args:
            field_type: Type of field
            value: Extracted value
            
        Returns:
            True if value is valid for the field type
        """
        if not value or len(value.strip()) < 2:
            return False
        
        validation_rules = {
            FieldType.INVOICE_NUMBER: lambda v: len(v) >= 4 and any(c.isalnum() for c in v),
            FieldType.INVOICE_DATE: lambda v: re.match(r'\d{1,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,4}', v),
            FieldType.ORDER_NUMBER: lambda v: len(v) >= 3 and any(c.isalnum() for c in v),
            FieldType.VENDOR_NAME: lambda v: len(v) >= 5 and any(c.isalpha() for c in v),
            FieldType.CUSTOMER_NAME: lambda v: len(v) >= 3 and any(c.isalpha() for c in v),
            FieldType.TOTAL_AMOUNT: lambda v: re.match(r'[\d,]+\.?\d*$', v.replace('$', '').strip()),
            FieldType.SUBTOTAL: lambda v: re.match(r'[\d,]+\.?\d*$', v.replace('$', '').strip()),
            FieldType.TAX_AMOUNT: lambda v: re.match(r'[\d,]+\.?\d*$', v.replace('$', '').strip()),
            FieldType.PAYMENT_TERMS: lambda v: len(v) >= 5,
        }
        
        validator = validation_rules.get(field_type, lambda v: True)
        return validator(value)
    
    def _calculate_field_confidence(
        self, 
        field_type: FieldType, 
        value: str, 
        pattern: str, 
        full_text: str, 
        start_pos: int, 
        end_pos: int
    ) -> float:
        """
        Calculate confidence score for a field match.
        
        Args:
            field_type: Type of field
            value: Extracted value
            pattern: Pattern that matched
            full_text: Full text being analyzed
            start_pos: Start position of match
            end_pos: End position of match
            
        Returns:
            Confidence score (0-100)
        """
        confidence = 50.0  # Base confidence
        
        # Context analysis (30% weight)
        context = self._extract_context(full_text, start_pos, end_pos, window=50)
        context_keywords = self.context_keywords.get(field_type, [])
        context_matches = sum(1 for kw in context_keywords if kw.lower() in context.lower())
        confidence += min(30, context_matches * 10)
        
        # Pattern specificity (25% weight)
        pattern_specificity = len(pattern) / 100  # Longer patterns are more specific
        confidence += min(25, pattern_specificity * 25)
        
        # Value quality (25% weight)
        value_quality = self._assess_value_quality(field_type, value)
        confidence += value_quality * 0.25
        
        # Position relevance (20% weight)
        position_score = self._assess_position_relevance(field_type, start_pos, len(full_text))
        confidence += position_score * 0.20
        
        return min(100, confidence)
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        """Extract context around a match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _assess_value_quality(self, field_type: FieldType, value: str) -> float:
        """Assess the quality of an extracted value (0-100)."""
        if not value:
            return 0
        
        score = 50  # Base score
        
        # Length appropriateness
        ideal_lengths = {
            FieldType.INVOICE_NUMBER: (6, 15),
            FieldType.INVOICE_DATE: (8, 12),
            FieldType.ORDER_NUMBER: (4, 20),
            FieldType.VENDOR_NAME: (10, 50),
            FieldType.CUSTOMER_NAME: (5, 50),
            FieldType.TOTAL_AMOUNT: (3, 15),
            FieldType.SUBTOTAL: (3, 15),
            FieldType.TAX_AMOUNT: (1, 10),
            FieldType.PAYMENT_TERMS: (5, 50),
        }
        
        min_len, max_len = ideal_lengths.get(field_type, (1, 100))
        if min_len <= len(value) <= max_len:
            score += 30
        
        # Character composition
        if field_type in [FieldType.TOTAL_AMOUNT, FieldType.SUBTOTAL, FieldType.TAX_AMOUNT]:
            if re.match(r'^\d+\.?\d*$', value.replace(',', '').replace('$', '')):
                score += 20
        elif field_type in [FieldType.VENDOR_NAME, FieldType.CUSTOMER_NAME]:
            if any(c.isupper() for c in value) and any(c.islower() for c in value):
                score += 20
        
        return min(100, score)
    
    def _assess_position_relevance(self, field_type: FieldType, position: int, text_length: int) -> float:
        """Assess how relevant the position is for the field type (0-100)."""
        relative_position = position / text_length if text_length > 0 else 0
        
        # Different field types are typically found in different parts of invoices
        preferred_positions = {
            FieldType.INVOICE_NUMBER: (0.0, 0.3),  # Top of document
            FieldType.INVOICE_DATE: (0.0, 0.3),   # Top of document
            FieldType.VENDOR_NAME: (0.0, 0.2),    # Very top
            FieldType.CUSTOMER_NAME: (0.1, 0.4),  # Upper portion
            FieldType.TOTAL_AMOUNT: (0.7, 1.0),   # Bottom of document
            FieldType.SUBTOTAL: (0.6, 0.9),       # Lower portion
            FieldType.TAX_AMOUNT: (0.6, 0.9),     # Lower portion
            FieldType.PAYMENT_TERMS: (0.8, 1.0),  # Bottom
        }
        
        min_pos, max_pos = preferred_positions.get(field_type, (0.0, 1.0))
        
        if min_pos <= relative_position <= max_pos:
            return 100
        else:
            # Calculate distance from preferred range
            if relative_position < min_pos:
                distance = min_pos - relative_position
            else:
                distance = relative_position - max_pos
            
            return max(0, 100 - (distance * 200))  # Penalty for being outside range
    
    def _remove_duplicate_matches(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Remove duplicate matches based on value similarity."""
        if not matches:
            return matches
        
        unique_matches = []
        seen_values = set()
        
        for match in matches:
            # Normalize value for comparison
            normalized_value = re.sub(r'\s+', ' ', match.value.lower().strip())
            
            if normalized_value not in seen_values:
                seen_values.add(normalized_value)
                unique_matches.append(match)
        
        return unique_matches
