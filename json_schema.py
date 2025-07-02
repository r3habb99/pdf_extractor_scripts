"""
Unified JSON Output Schema

This module defines the consistent JSON schema for PDF extraction results,
ensuring uniform output regardless of the extraction method used.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class InvoiceHeader:
    """Invoice header information."""
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    order_number: Optional[str] = None
    customer_po: Optional[str] = None
    due_date: Optional[str] = None


@dataclass
class VendorInfo:
    """Vendor/supplier information."""
    company_name: Optional[str] = None
    address: Optional[str] = None
    city_state_zip: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    tax_id: Optional[str] = None


@dataclass
class CustomerInfo:
    """Customer information."""
    sold_to_id: Optional[str] = None
    ship_to_id: Optional[str] = None
    company_name: Optional[str] = None
    address: Optional[str] = None
    suite: Optional[str] = None
    city_state_zip: Optional[str] = None
    contact_person: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


@dataclass
class LineItem:
    """Individual line item with enhanced pricing breakdown."""
    line_number: Optional[str] = None
    plant: Optional[str] = None
    item_code: Optional[str] = None
    description: Optional[str] = None
    quantity_ordered: Optional[Union[int, str]] = None
    quantity_shipped: Optional[Union[int, str]] = None
    quantity_backordered: Optional[Union[int, str]] = None  # BO field
    unit_of_measure: Optional[str] = None
    list_price: Optional[Union[float, str]] = None
    discount_percent: Optional[Union[float, str]] = None
    unit_price: Optional[Union[float, str]] = None
    extended_amount: Optional[Union[float, str]] = None
    tax_amount: Optional[Union[float, str]] = None
    # Enhanced pricing breakdown structure
    pricing_breakdown: Optional[Dict[str, Any]] = None
    product_details: Optional[Dict[str, Any]] = None


@dataclass
class Totals:
    """Invoice totals."""
    subtotal: Optional[Union[float, str]] = None
    discount_amount: Optional[Union[float, str]] = None
    total_sale: Optional[Union[float, str]] = None
    tax: Optional[Union[float, str]] = None
    shipping: Optional[Union[float, str]] = None
    invoice_total: Optional[Union[float, str]] = None
    amount_due: Optional[Union[float, str]] = None


@dataclass
class PaymentTerms:
    """Payment terms and conditions."""
    terms: Optional[str] = None
    due_date: Optional[str] = None
    discount_date: Optional[str] = None
    discount_percent: Optional[Union[float, str]] = None
    net_days: Optional[int] = None


@dataclass
class ShippingInfo:
    """Shipping and delivery information."""
    tracking_number: Optional[str] = None
    carrier: Optional[str] = None
    ship_from: Optional[str] = None
    ship_to: Optional[str] = None
    ship_date: Optional[str] = None
    delivery_date: Optional[str] = None
    shipping_method: Optional[str] = None
    freight_charge: Optional[Union[float, str]] = None


@dataclass
class ProcessingMetadata:
    """Metadata about the processing."""
    pdf_path: str
    extraction_method: str  # 'text_extraction', 'ocr', 'hybrid'
    processor: str
    total_pages: int
    processing_timestamp: str
    confidence_score: Optional[float] = None
    ocr_engines_used: Optional[List[str]] = None
    processing_time_seconds: Optional[float] = None


class UnifiedJSONSchema:
    """
    Unified JSON schema for PDF extraction results.
    Ensures consistent output structure regardless of extraction method.
    """
    
    def __init__(self):
        """Initialize the schema."""
        pass
    
    def create_structured_output(
        self,
        invoice_header: Optional[Dict[str, Any]] = None,
        vendor_info: Optional[Dict[str, Any]] = None,
        customer_info: Optional[Dict[str, Any]] = None,
        line_items: Optional[List[Dict[str, Any]]] = None,
        totals: Optional[Dict[str, Any]] = None,
        payment_terms: Optional[Dict[str, Any]] = None,
        shipping_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a structured output following the unified schema.
        
        Args:
            invoice_header: Invoice header information
            vendor_info: Vendor information
            customer_info: Customer information
            line_items: List of line items
            totals: Invoice totals
            payment_terms: Payment terms
            shipping_info: Shipping information
            metadata: Processing metadata
            **kwargs: Additional fields
            
        Returns:
            Structured dictionary following the unified schema
        """
        # Create structured objects
        header = InvoiceHeader(**(invoice_header or {}))
        vendor = VendorInfo(**(vendor_info or {}))
        customer = CustomerInfo(**(customer_info or {}))
        
        # Process line items
        processed_line_items = []
        for item in (line_items or []):
            processed_line_items.append(asdict(LineItem(**item)))
        
        totals_obj = Totals(**(totals or {}))
        terms_obj = PaymentTerms(**(payment_terms or {}))
        shipping_obj = ShippingInfo(**(shipping_info or {}))
        
        # Add processing timestamp if not provided
        if metadata:
            if 'processing_timestamp' not in metadata:
                metadata['processing_timestamp'] = datetime.now().isoformat()
            metadata_obj = ProcessingMetadata(**metadata)
        else:
            metadata_obj = ProcessingMetadata(
                pdf_path="",
                extraction_method="unknown",
                processor="unknown",
                total_pages=0,
                processing_timestamp=datetime.now().isoformat()
            )
        
        # Create the unified structure
        structured_output = {
            "invoice_header": asdict(header),
            "vendor_info": asdict(vendor),
            "customer_info": asdict(customer),
            "line_items": processed_line_items,
            "totals": asdict(totals_obj),
            "payment_terms": asdict(terms_obj),
            "shipping_info": asdict(shipping_obj),
            "metadata": asdict(metadata_obj)
        }
        
        # Add any additional fields
        for key, value in kwargs.items():
            if key not in structured_output:
                structured_output[key] = value
        
        # Clean up None values if desired (optional)
        structured_output = self._clean_none_values(structured_output)
        
        return structured_output
    
    def _clean_none_values(self, data: Any, remove_empty: bool = False) -> Any:
        """
        Recursively remove None values from the data structure.
        
        Args:
            data: Data to clean
            remove_empty: Whether to remove empty strings and lists
            
        Returns:
            Cleaned data structure
        """
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                cleaned_value = self._clean_none_values(value, remove_empty)
                
                # Keep the value if it's not None
                if cleaned_value is not None:
                    # Optionally remove empty strings and lists
                    if remove_empty:
                        if cleaned_value != "" and cleaned_value != []:
                            cleaned[key] = cleaned_value
                    else:
                        cleaned[key] = cleaned_value
            return cleaned
        
        elif isinstance(data, list):
            return [self._clean_none_values(item, remove_empty) for item in data if item is not None]
        
        else:
            return data
    
    def validate_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the data follows the expected schema.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        required_sections = [
            "invoice_header", "vendor_info", "customer_info", 
            "line_items", "totals", "payment_terms", 
            "shipping_info", "metadata"
        ]
        
        # Check for required sections
        for section in required_sections:
            if section not in data:
                validation_result["errors"].append(f"Missing required section: {section}")
                validation_result["is_valid"] = False
        
        # Check metadata requirements
        if "metadata" in data:
            required_metadata = ["pdf_path", "extraction_method", "processor", "total_pages"]
            for field in required_metadata:
                if field not in data["metadata"]:
                    validation_result["warnings"].append(f"Missing metadata field: {field}")
        
        # Check line items structure
        if "line_items" in data and isinstance(data["line_items"], list):
            for i, item in enumerate(data["line_items"]):
                if not isinstance(item, dict):
                    validation_result["errors"].append(f"Line item {i} is not a dictionary")
                    validation_result["is_valid"] = False
        
        return validation_result
    
    def to_json(self, data: Dict[str, Any], indent: int = 2) -> str:
        """
        Convert structured data to JSON string.
        
        Args:
            data: Structured data
            indent: JSON indentation
            
        Returns:
            JSON string
        """
        return json.dumps(data, indent=indent, ensure_ascii=False)
    
    def from_json(self, json_str: str) -> Dict[str, Any]:
        """
        Parse JSON string to structured data.
        
        Args:
            json_str: JSON string
            
        Returns:
            Structured data dictionary
        """
        return json.loads(json_str)


# Convenience function
def create_unified_output(**kwargs) -> Dict[str, Any]:
    """
    Convenience function to create unified output.
    
    Args:
        **kwargs: Arguments for create_structured_output
        
    Returns:
        Structured output dictionary
    """
    schema = UnifiedJSONSchema()
    return schema.create_structured_output(**kwargs)


# Example usage and schema documentation
SCHEMA_EXAMPLE = {
    "invoice_header": {
        "invoice_number": "01793287",
        "invoice_date": "6/13/25",
        "order_number": "F3GU7A-00",
        "customer_po": "370555-001",
        "due_date": "8/12/25"
    },
    "vendor_info": {
        "company_name": "Ceco Door",
        "address": "9159 TELECOM DR.",
        "city_state_zip": "MILAN, TN 38358",
        "phone": "(888) 264-7474"
    },
    "customer_info": {
        "sold_to_id": "18010812",
        "ship_to_id": "18010812",
        "company_name": "COOK & BOARDMAN",
        "address": "1250 IMESON PARK BLVD",
        "suite": "STE 419",
        "city_state_zip": "JACKSONVILLE FL 32218"
    },
    "line_items": [
        {
            "line_number": "001",
            "plant": "572",
            "item_code": "FR3PC",
            "description": "3 PIECE FRAME",
            "quantity_ordered": 1,
            "quantity_shipped": 1,
            "list_price": 829.0,
            "discount_percent": 54.871,
            "unit_price": 374.11,
            "extended_amount": 374.12
        }
    ],
    "totals": {
        "total_sale": "2,244.78",
        "tax": "0.00",
        "invoice_total": "2,244.78"
    },
    "payment_terms": {
        "terms": "2% 30 DAYS, NET 60",
        "due_date": "8/12/25"
    },
    "shipping_info": {
        "carrier": "CUSTOMER PICKUP"
    },
    "metadata": {
        "pdf_path": "/path/to/invoice.pdf",
        "extraction_method": "text_extraction",
        "processor": "TextPDFProcessor",
        "total_pages": 7,
        "processing_timestamp": "2025-06-25T10:30:00",
        "confidence_score": 95.5
    }
}
