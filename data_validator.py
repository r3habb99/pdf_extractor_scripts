"""
Data Validation and Post-Processing Module

This module provides comprehensive validation and post-processing for extracted PDF data
to ensure completeness and quality of the extracted information.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    completeness_score: float
    missing_fields: List[str]
    empty_fields: List[str]
    warnings: List[str]
    suggestions: List[str]


class DataValidator:
    """
    Comprehensive data validator for extracted PDF information.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.required_fields = {
            'invoice_header': ['invoice_number', 'invoice_date'],
            'vendor_info': ['company_name'],
            'customer_info': ['company_name'],
            'line_items': ['line_number', 'description', 'quantity_ordered', 'unit_price'],
            'totals': [],  # Optional but recommended
            'payment_terms': [],  # Optional
            'shipping_info': [],  # Optional
            'metadata': ['pdf_path', 'extraction_method', 'processor']
        }
        
        self.recommended_fields = {
            'invoice_header': ['due_date', 'order_number'],
            'vendor_info': ['address', 'phone'],
            'customer_info': ['address'],
            'line_items': ['extended_amount', 'item_code'],
            'totals': ['invoice_total', 'subtotal'],
            'payment_terms': ['terms', 'due_date'],
            'shipping_info': ['tracking_number'],
            'metadata': ['confidence_score', 'processing_timestamp']
        }
    
    def validate_extracted_data(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate extracted data for completeness and quality.
        
        Args:
            data: Extracted data dictionary
            
        Returns:
            ValidationResult with validation details
        """
        missing_fields = []
        empty_fields = []
        warnings = []
        suggestions = []
        
        # Check required fields
        for section, fields in self.required_fields.items():
            section_data = data.get(section, {})
            
            if not section_data and fields:
                missing_fields.append(f"Section '{section}' is missing")
                continue
            
            for field in fields:
                if section == 'line_items':
                    # Special handling for line items
                    if not section_data or len(section_data) == 0:
                        missing_fields.append(f"No line items found")
                    else:
                        self._validate_line_items(section_data, field, empty_fields, warnings)
                else:
                    if field not in section_data:
                        missing_fields.append(f"{section}.{field}")
                    elif not self._is_field_populated(section_data[field]):
                        empty_fields.append(f"{section}.{field}")
        
        # Check recommended fields and provide suggestions
        for section, fields in self.recommended_fields.items():
            section_data = data.get(section, {})
            
            for field in fields:
                if section == 'line_items':
                    if section_data and len(section_data) > 0:
                        missing_count = sum(
                            1 for item in section_data 
                            if not self._is_field_populated(item.get(field))
                        )
                        if missing_count > len(section_data) * 0.5:  # More than 50% missing
                            suggestions.append(f"Consider improving extraction of line_items.{field}")
                else:
                    if field not in section_data or not self._is_field_populated(section_data[field]):
                        suggestions.append(f"Consider adding {section}.{field} for better completeness")
        
        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(data)
        
        # Determine if validation passes
        is_valid = (
            len(missing_fields) == 0 and 
            len(empty_fields) <= 2 and  # Allow up to 2 empty optional fields
            completeness_score >= 60.0  # Minimum 60% completeness
        )
        
        return ValidationResult(
            is_valid=is_valid,
            completeness_score=completeness_score,
            missing_fields=missing_fields,
            empty_fields=empty_fields,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_line_items(self, line_items: List[Dict[str, Any]], field: str, 
                           empty_fields: List[str], warnings: List[str]) -> None:
        """Validate line items for a specific field."""
        empty_count = 0
        
        for i, item in enumerate(line_items):
            if field not in item or not self._is_field_populated(item[field]):
                empty_count += 1
                empty_fields.append(f"line_items[{i}].{field}")
        
        if empty_count > len(line_items) * 0.3:  # More than 30% empty
            warnings.append(f"High number of empty {field} in line items ({empty_count}/{len(line_items)})")
    
    def _is_field_populated(self, value: Any) -> bool:
        """Check if a field has meaningful content."""
        if value is None:
            return False
        
        if isinstance(value, str):
            return len(value.strip()) > 0
        
        if isinstance(value, (int, float)):
            return value != 0
        
        if isinstance(value, (list, dict)):
            return len(value) > 0
        
        return True
    
    def _calculate_completeness_score(self, data: Dict[str, Any]) -> float:
        """Calculate a completeness score for the extracted data."""
        total_score = 0.0
        max_score = 100.0
        
        # Invoice header (25 points)
        header = data.get('invoice_header', {})
        if self._is_field_populated(header.get('invoice_number')):
            total_score += 15
        if self._is_field_populated(header.get('invoice_date')):
            total_score += 10
        
        # Vendor info (15 points)
        vendor = data.get('vendor_info', {})
        if self._is_field_populated(vendor.get('company_name')):
            total_score += 10
        if self._is_field_populated(vendor.get('address')):
            total_score += 5
        
        # Customer info (15 points)
        customer = data.get('customer_info', {})
        if self._is_field_populated(customer.get('company_name')):
            total_score += 10
        if self._is_field_populated(customer.get('address')):
            total_score += 5
        
        # Line items (35 points - most important)
        line_items = data.get('line_items', [])
        if line_items:
            base_score = 20
            total_score += base_score
            
            # Quality bonus for complete line items
            complete_items = sum(
                1 for item in line_items
                if all(self._is_field_populated(item.get(field)) 
                      for field in ['line_number', 'description', 'quantity_ordered', 'unit_price'])
            )
            
            if len(line_items) > 0:
                quality_ratio = complete_items / len(line_items)
                total_score += 15 * quality_ratio
        
        # Totals (10 points)
        totals = data.get('totals', {})
        if totals and any(self._is_field_populated(v) for v in totals.values()):
            total_score += 10
        
        return min(total_score, max_score)
    
    def enhance_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance extracted data by filling in missing fields where possible.
        
        Args:
            data: Original extracted data
            
        Returns:
            Enhanced data with filled fields
        """
        enhanced_data = data.copy()
        
        # Enhance line items
        line_items = enhanced_data.get('line_items', [])
        for item in line_items:
            self._enhance_line_item(item)
        
        # Calculate missing totals if possible
        self._calculate_missing_totals(enhanced_data, line_items)
        
        # Enhance dates
        self._enhance_dates(enhanced_data)
        
        # Clean up empty values
        self._clean_empty_values(enhanced_data)
        
        return enhanced_data
    
    def _enhance_line_item(self, item: Dict[str, Any]) -> None:
        """Enhance a single line item with calculated fields."""
        # Calculate extended amount if missing
        if (not self._is_field_populated(item.get('extended_amount')) and
            self._is_field_populated(item.get('quantity_ordered')) and
            self._is_field_populated(item.get('unit_price'))):
            
            try:
                qty = float(item['quantity_ordered'])
                price = float(item['unit_price'])
                item['extended_amount'] = round(qty * price, 2)
            except (ValueError, TypeError):
                pass
        
        # Set quantity_shipped to quantity_ordered if missing
        if (not self._is_field_populated(item.get('quantity_shipped')) and
            self._is_field_populated(item.get('quantity_ordered'))):
            item['quantity_shipped'] = item['quantity_ordered']
    
    def _calculate_missing_totals(self, data: Dict[str, Any], line_items: List[Dict[str, Any]]) -> None:
        """Calculate missing totals from line items."""
        totals = data.setdefault('totals', {})
        
        if line_items and not self._is_field_populated(totals.get('subtotal')):
            try:
                subtotal = sum(
                    float(item.get('extended_amount', 0))
                    for item in line_items
                    if self._is_field_populated(item.get('extended_amount'))
                )
                if subtotal > 0:
                    totals['subtotal'] = f"{subtotal:.2f}"
            except (ValueError, TypeError):
                pass
    
    def _enhance_dates(self, data: Dict[str, Any]) -> None:
        """Enhance and standardize date formats."""
        # This could include date format standardization
        # For now, just ensure dates are present
        pass
    
    def _clean_empty_values(self, data: Dict[str, Any]) -> None:
        """Remove or replace empty values with meaningful defaults."""
        for section_name, section_data in data.items():
            if isinstance(section_data, dict):
                # Remove empty string values
                keys_to_update = []
                for key, value in section_data.items():
                    if isinstance(value, str) and value.strip() == "":
                        keys_to_update.append(key)
                
                for key in keys_to_update:
                    del section_data[key]
            
            elif isinstance(section_data, list):
                # Clean line items
                for item in section_data:
                    if isinstance(item, dict):
                        keys_to_update = []
                        for key, value in item.items():
                            if isinstance(value, str) and value.strip() == "":
                                keys_to_update.append(key)
                        
                        for key in keys_to_update:
                            del item[key]
