"""
Enhanced Text-Selectable PDF Processor

This module handles extraction from PDFs that contain selectable text,
preserving layout and structure while extracting data in a structured format.
Features dynamic pattern matching, confidence scoring, and adaptive extraction.
"""

import pdfplumber
import PyPDF2
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextElement:
    """Represents a text element with position and formatting information."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    fontname: Optional[str] = None
    fontsize: Optional[float] = None
    page_number: int = 0


@dataclass
class ExtractionResult:
    """Represents an extraction result with confidence scoring."""
    value: Any
    confidence: float
    pattern_used: Optional[str] = None
    method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldPattern:
    """Represents a field extraction pattern with metadata."""
    pattern: str
    confidence_weight: float = 1.0
    description: str = ""
    vendor_specific: bool = False
    required_context: Optional[List[str]] = None


@dataclass
class VendorConfig:
    """Configuration for vendor-specific extraction patterns."""
    name: str
    patterns: Dict[str, List[FieldPattern]]
    table_indicators: List[str]
    confidence_thresholds: Dict[str, float]
    field_mappings: Dict[str, List[str]]
    validation_rules: Dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """Manages extraction patterns and vendor-specific configurations."""

    def __init__(self):
        self.vendor_configs = {}
        self.default_config = None
        self._load_configurations()

    def _load_configurations(self):
        """Load vendor configurations from files or create defaults."""
        try:
            # Try to load from config files first
            config_dir = Path(__file__).parent / "config"
            if config_dir.exists():
                self._load_from_files(config_dir)
            else:
                self._create_default_configurations()
        except Exception as e:
            logger.warning(f"Failed to load configurations: {e}. Using defaults.")
            self._create_default_configurations()

    def _load_from_files(self, config_dir: Path):
        """
        Load vendor configurations from YAML or JSON files in the given directory.
        This is a stub implementation; you can expand it to actually load files as needed.
        """
        # Example: Load all YAML files in the config_dir
        for config_file in config_dir.glob("*.yaml"):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
                    # You would need to parse config_data into VendorConfig objects here
                    # For now, just log the loading
                    logger.info(f"Loaded vendor config from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")

    def _create_default_configurations(self):
        """Create default configurations for known vendors."""
        # CECO Configuration
        ceco_patterns = {
            "invoice_number": [
                FieldPattern(r"^\.\s*([0-9]{8})\s", 0.9, "CECO dot pattern"),
                FieldPattern(r"^\.?\s*([0-9]{8})\s+\d{1,2}\/\d{1,2}\/\d{2,4}", 0.8, "CECO number with date"),
                FieldPattern(r"invoice\s*number\s*:?\s*([A-Z0-9\-]{6,15})", 0.6, "Generic invoice number"),
                FieldPattern(r"\b([0-9]{8})\b", 0.4, "8-digit number fallback")
            ],
            "invoice_date": [
                FieldPattern(r"^\.\s*[0-9]{8}\s+(\d{1,2}\/\d{1,2}\/\d{2,4})", 0.9, "CECO date after number"),
                FieldPattern(r"invoice\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", 0.7, "Generic invoice date"),
                FieldPattern(r"\b(\d{1,2}\/\d{1,2}\/\d{2,4})\b", 0.5, "Date pattern fallback")
            ],
            "order_number": [
                FieldPattern(r"\b([A-Z0-9]{6}-[0-9]{2})\b", 0.8, "CECO order format"),
                FieldPattern(r"order\s*no\s*:?\s*([A-Z0-9\-]{3,20})", 0.6, "Generic order number"),
                FieldPattern(r"order\s*number\s*:?\s*([A-Z0-9\-]{3,20})", 0.6, "Generic order number alt")
            ],
            "customer_po": [
                FieldPattern(r"\b([0-9]{6}-[0-9]{3})\b", 0.7, "Hyphenated PO format"),
                FieldPattern(r"(?<!01)\b([0-9]{7})\b(?!\s*\d{1,2}\/)", 0.6, "7-digit PO"),
                FieldPattern(r"customer\s*po\s*:?\s*([A-Z0-9\-]{3,20})", 0.5, "Generic customer PO"),
                FieldPattern(r"po\s*#?\s*:?\s*([A-Z0-9\-]{3,20})", 0.5, "Generic PO number")
            ],
            "company_name": [
                FieldPattern(r"(Ceco\s+Door\s+Products)", 0.9, "CECO company name"),
                FieldPattern(r"(CECO\s+DOOR\s+PRODUCTS)", 0.9, "CECO company name caps"),
                FieldPattern(r"([A-Z][A-Za-z\s&]+(?:Inc|LLC|Corp|Company|Co\.|Ltd|Products|Services|Solutions)\.?)", 0.6, "Company with suffix"),
                FieldPattern(r"([A-Z][A-Za-z\s&]{5,50})", 0.4, "Generic company name")
            ],
            "address": [
                FieldPattern(r"(\d+\s+TELECOM\s+DR\.)", 0.8, "CECO TELECOM address"),
                FieldPattern(r"(\d+\s+[A-Z][A-Za-z\s]+(?:ST|STREET|AVE|AVENUE|BLVD|BOULEVARD|DR|DRIVE|RD|ROAD)\.?)", 0.7, "Street address"),
                FieldPattern(r"(\d+\s+[A-Z][A-Za-z\s]{5,30})", 0.5, "Generic address")
            ],
            "city_state_zip": [
                FieldPattern(r"(MILAN),?\s+(TN)\s+(\d{5})", 0.8, "CECO Milan TN"),
                FieldPattern(r"([A-Z][A-Za-z\s]+),?\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)", 0.8, "City, State ZIP"),
                FieldPattern(r"([A-Z][A-Za-z\s]+)\s+([A-Z]{2})\s+(\d{5})", 0.7, "City State ZIP")
            ],
            "phone": [
                FieldPattern(r"(\(888\)\s+264-7474)", 0.9, "CECO phone number"),
                FieldPattern(r"(\(\d{3}\)\s+\d{3}-\d{4})", 0.8, "Phone with parentheses"),
                FieldPattern(r"(\d{3})-(\d{3})-(\d{4})", 0.7, "Phone with dashes"),
                FieldPattern(r"(\d{3})\.(\d{3})\.(\d{4})", 0.6, "Phone with dots")
            ]
        }

        self.vendor_configs['ceco'] = VendorConfig(
            name="CECO",
            patterns=ceco_patterns,
            table_indicators=["line", "plant", "item", "qty", "description"],
            confidence_thresholds={"invoice_number": 0.8, "invoice_date": 0.7, "line_items": 0.6},
            field_mappings={
                "line_number": ["line", "item", "#", "no"],
                "item_code": ["item", "code", "part", "sku"],
                "quantity_ordered": ["qty", "quantity", "ord", "ordered"],
                "quantity_shipped": ["ship", "shipped", "qty ship"],
                "unit_price": ["unit", "price", "each"],
                "extended_amount": ["amount", "total", "extended", "ext"]
            }
        )

        # Schlage Configuration
        schlage_patterns = {
            "invoice_number": [
                FieldPattern(r"invoice\s*number\s*:?\s*([A-Z0-9\-]{6,20})", 0.8, "Schlage invoice number"),
                FieldPattern(r"invoice\s*#\s*:?\s*([A-Z0-9\-]{6,20})", 0.7, "Invoice # format"),
                FieldPattern(r"\b([0-9]{6}-[0-9]{3})\b", 0.6, "Hyphenated invoice format")
            ],
            "invoice_date": [
                FieldPattern(r"invoice\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", 0.8, "Schlage invoice date"),
                FieldPattern(r"date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", 0.6, "Generic date"),
                FieldPattern(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b", 0.4, "Date pattern")
            ],
            "company_name": [
                FieldPattern(r"(Schlage)", 0.9, "Schlage company name"),
                FieldPattern(r"(SCHLAGE)", 0.9, "Schlage company name caps"),
                FieldPattern(r"(Allegion)", 0.8, "Allegion parent company"),
                FieldPattern(r"([A-Z][A-Za-z\s&]+(?:Inc|LLC|Corp|Company|Co\.|Ltd|Products|Services|Solutions)\.?)", 0.6, "Company with suffix")
            ],
            "order_number": [
                FieldPattern(r"order\s*number\s*:?\s*([A-Z0-9\-]{6,20})", 0.7, "Schlage order number"),
                FieldPattern(r"order\s*no\s*:?\s*([A-Z0-9\-]{6,20})", 0.6, "Order no format")
            ],
            "customer_po": [
                FieldPattern(r"customer\s*po\s*:?\s*([A-Z0-9\-]{3,20})", 0.7, "Schlage customer PO"),
                FieldPattern(r"po\s*#?\s*:?\s*([A-Z0-9\-]{3,20})", 0.6, "PO number")
            ]
        }

        self.vendor_configs['schlage'] = VendorConfig(
            name="Schlage",
            patterns=schlage_patterns,
            table_indicators=["item", "description", "qty", "price", "amount"],
            confidence_thresholds={"invoice_number": 0.7, "invoice_date": 0.7, "line_items": 0.6},
            field_mappings={
                "line_number": ["line", "item", "#", "no", "row"],
                "item_code": ["item", "code", "part", "sku", "product"],
                "description": ["description", "desc", "product", "item"],
                "quantity": ["qty", "quantity", "amount", "count"],
                "unit_price": ["unit", "price", "each", "rate"],
                "total": ["total", "amount", "extended", "sum"]
            }
        )

        # Steelcraft Configuration
        steelcraft_patterns = {
            "invoice_number": [
                FieldPattern(r"invoice\s*number\s*:?\s*([A-Z0-9\-]{6,20})", 0.8, "Steelcraft invoice number"),
                FieldPattern(r"invoice\s*#\s*:?\s*([A-Z0-9\-]{6,20})", 0.7, "Invoice # format"),
                FieldPattern(r"\b([A-Z0-9]{8,15})\b", 0.5, "Alphanumeric invoice")
            ],
            "invoice_date": [
                FieldPattern(r"invoice\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", 0.8, "Steelcraft invoice date"),
                FieldPattern(r"date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", 0.6, "Generic date"),
                FieldPattern(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b", 0.4, "Date pattern")
            ],
            "company_name": [
                FieldPattern(r"(Steelcraft)", 0.9, "Steelcraft company name"),
                FieldPattern(r"(STEELCRAFT)", 0.9, "Steelcraft company name caps"),
                FieldPattern(r"(Allegion)", 0.8, "Allegion parent company"),
                FieldPattern(r"([A-Z][A-Za-z\s&]+(?:Inc|LLC|Corp|Company|Co\.|Ltd|Products|Services|Solutions)\.?)", 0.6, "Company with suffix")
            ],
            "order_number": [
                FieldPattern(r"order\s*number\s*:?\s*([A-Z0-9\-]{6,20})", 0.7, "Steelcraft order number"),
                FieldPattern(r"order\s*no\s*:?\s*([A-Z0-9\-]{6,20})", 0.6, "Order no format")
            ],
            "customer_po": [
                FieldPattern(r"customer\s*po\s*:?\s*([A-Z0-9\-]{3,20})", 0.7, "Steelcraft customer PO"),
                FieldPattern(r"po\s*#?\s*:?\s*([A-Z0-9\-]{3,20})", 0.6, "PO number")
            ]
        }

        self.vendor_configs['steelcraft'] = VendorConfig(
            name="Steelcraft",
            patterns=steelcraft_patterns,
            table_indicators=["item", "description", "qty", "price", "amount"],
            confidence_thresholds={"invoice_number": 0.7, "invoice_date": 0.7, "line_items": 0.6},
            field_mappings={
                "line_number": ["line", "item", "#", "no", "row"],
                "item_code": ["item", "code", "part", "sku", "product"],
                "description": ["description", "desc", "product", "item"],
                "quantity": ["qty", "quantity", "amount", "count"],
                "unit_price": ["unit", "price", "each", "rate"],
                "total": ["total", "amount", "extended", "sum"]
            }
        )

        # Generic Configuration
        generic_patterns = {
            "invoice_number": [
                FieldPattern(r"invoice\s*number\s*:?\s*([A-Z0-9\-]{3,20})", 0.7, "Generic invoice number"),
                FieldPattern(r"invoice\s*#\s*:?\s*([A-Z0-9\-]{3,20})", 0.6, "Invoice # format"),
                FieldPattern(r"\b([A-Z0-9]{6,15})\b", 0.3, "Alphanumeric fallback")
            ],
            "invoice_date": [
                FieldPattern(r"invoice\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", 0.7, "Generic invoice date"),
                FieldPattern(r"date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", 0.5, "Generic date"),
                FieldPattern(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b", 0.3, "Date pattern")
            ],
            "company_name": [
                FieldPattern(r"([A-Z][A-Za-z\s&]+(?:Inc|LLC|Corp|Company|Co\.|Ltd|Products|Services|Solutions)\.?)", 0.6, "Company with suffix"),
                FieldPattern(r"([A-Z][A-Za-z\s&]{5,50})", 0.4, "Generic company name")
            ],
            "address": [
                FieldPattern(r"(\d+\s+[A-Z][A-Za-z\s]+(?:ST|STREET|AVE|AVENUE|BLVD|BOULEVARD|DR|DRIVE|RD|ROAD)\.?)", 0.7, "Street address"),
                FieldPattern(r"(\d+\s+[A-Z][A-Za-z\s]{5,30})", 0.5, "Generic address")
            ],
            "city_state_zip": [
                FieldPattern(r"([A-Z][A-Za-z\s]+),?\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)", 0.8, "City, State ZIP"),
                FieldPattern(r"([A-Z][A-Za-z\s]+)\s+([A-Z]{2})\s+(\d{5})", 0.7, "City State ZIP")
            ],
            "phone": [
                FieldPattern(r"(\(\d{3}\)\s+\d{3}-\d{4})", 0.8, "Phone with parentheses"),
                FieldPattern(r"(\d{3})-(\d{3})-(\d{4})", 0.7, "Phone with dashes"),
                FieldPattern(r"(\d{3})\.(\d{3})\.(\d{4})", 0.6, "Phone with dots")
            ],
            "email": [
                FieldPattern(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", 0.9, "Email address")
            ],
            "discount_amount": [
                FieldPattern(r"discount\s*of\s*\$\s*([\d,]+\.?\d*)", 0.8, "Discount of amount"),
                FieldPattern(r"discount\s*amount\s*:?\s*\$?([\d,]+\.?\d*)", 0.7, "Discount amount"),
                FieldPattern(r"total\s*discount\s*:?\s*\$?([\d,]+\.?\d*)", 0.6, "Total discount")
            ],
            "total_sale": [
                FieldPattern(r"total\s*sale\s*:?\s*\$?\s*([\d,]+\.?\d*)", 0.8, "Total sale"),
                FieldPattern(r"subtotal\s*:?\s*\$?([\d,]+\.?\d*)", 0.7, "Subtotal"),
                FieldPattern(r"sub\s*total\s*:?\s*\$?([\d,]+\.?\d*)", 0.6, "Sub total")
            ],
            "tax": [
                FieldPattern(r"tax\s*:?\s*\$?\s*([\d,]+\.?\d*)", 0.8, "Tax"),
                FieldPattern(r"sales\s*tax\s*:?\s*\$?([\d,]+\.?\d*)", 0.7, "Sales tax")
            ],
            "invoice_total": [
                FieldPattern(r"invoice\s*total\s*\(USD\)\s*:?\s*\$?\s*([\d,]+\.?\d*)", 0.9, "Invoice total USD"),
                FieldPattern(r"invoice\s*total\s*:?\s*\$?([\d,]+\.?\d*)", 0.8, "Invoice total"),
                FieldPattern(r"total\s*due\s*:?\s*\$?([\d,]+\.?\d*)", 0.7, "Total due"),
                FieldPattern(r"amount\s*due\s*:?\s*\$?([\d,]+\.?\d*)", 0.6, "Amount due")
            ],
            "freight": [
                FieldPattern(r"freight\s*:?\s*\$?([\d,]+\.?\d*)", 0.8, "Freight"),
                FieldPattern(r"shipping\s*:?\s*\$?([\d,]+\.?\d*)", 0.7, "Shipping")
            ],
            "terms": [
                FieldPattern(r"(\d+%\s+\d+\s+DAYS,\s+NET\s+\d+)", 0.8, "Payment terms format"),
                FieldPattern(r"(NET\s+\d+)", 0.6, "Net terms")
            ],
            "due_date": [
                FieldPattern(r"payable\s*on\s*(\d{1,2}\/\d{1,2}\/\d{2,4})", 0.8, "Payable on date"),
                FieldPattern(r"due\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", 0.7, "Due date")
            ],
            "discount_date": [
                FieldPattern(r"received\s*on\s*or\s*before\s*(\d{1,2}\/\d{1,2}\/\d{2,4})", 0.8, "Discount date"),
                FieldPattern(r"discount\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", 0.7, "Discount date label")
            ],
            "tracking_number": [
                FieldPattern(r"shipment\s*tracking\s*numb[er]*\s*:?\s*([A-Z0-9]+)", 0.8, "Shipment tracking"),
                FieldPattern(r"tracking\s*numb[er]*\s*:?\s*([A-Z0-9]+)", 0.7, "Tracking number")
            ],
            "carrier": [
                FieldPattern(r"carrier\s*:?\s*(CUSTOMER\s+PICKUP[^\\n]*)", 0.8, "Customer pickup"),
                FieldPattern(r"([A-Z\s]+LOGISTICS)(?:\s|$)", 0.7, "Logistics carrier"),
                FieldPattern(r"([A-Z\s]+(?:LOGISTICS|TRANSPORT|SHIPPING|EXPRESS))(?:\s|$)", 0.6, "Transport carrier")
            ],
            "ship_from": [
                FieldPattern(r"order\s*shipped\s*from\s*(\d+\s*-\s*[A-Z\s]+(?:MANUFACTURING|PLANT|FACILITY)?)", 0.8, "Shipped from"),
                FieldPattern(r"shipped\s*from\s*(\d+\s*-\s*[^\\n]+)", 0.6, "Shipped from generic")
            ],
            "shipping_method": [
                FieldPattern(r"(PREPAID\s+3RD\s+PARTY)", 0.8, "Prepaid 3rd party"),
                FieldPattern(r"(F\.O\.B\.\s+SHIP\s+POINT)", 0.7, "FOB ship point"),
                FieldPattern(r"(CUSTOMER\s+PICKUP)", 0.6, "Customer pickup")
            ],
            "freight_charge": [
                FieldPattern(r"freight\s*charge\s*:?\s*\$?\s*([\d,]+\.?\d*)", 0.8, "Freight charge"),
                FieldPattern(r"freight\s*:?\s*\$?\s*([\d,]+\.?\d*)", 0.7, "Freight amount"),
                FieldPattern(r"shipping\s*charge\s*:?\s*\$?\s*([\d,]+\.?\d*)", 0.6, "Shipping charge")
            ]
        }

        self.vendor_configs['generic'] = VendorConfig(
            name="Generic",
            patterns=generic_patterns,
            table_indicators=["item", "description", "qty", "price", "amount"],
            confidence_thresholds={"invoice_number": 0.6, "invoice_date": 0.6, "line_items": 0.5},
            field_mappings={
                "line_number": ["line", "item", "#", "no", "row"],
                "item_code": ["item", "code", "part", "sku", "product"],
                "description": ["description", "desc", "product", "item"],
                "quantity": ["qty", "quantity", "amount", "count"],
                "unit_price": ["unit", "price", "each", "rate"],
                "total": ["total", "amount", "extended", "sum"]
            }
        )

        self.default_config = self.vendor_configs['generic']

    def get_config(self, vendor_type: str = 'generic') -> VendorConfig:
        """Get configuration for a specific vendor."""
        return self.vendor_configs.get(vendor_type.lower(), self.default_config)

    def get_patterns(self, vendor_type: str, field_name: str) -> List[FieldPattern]:
        """Get patterns for a specific field and vendor."""
        config = self.get_config(vendor_type)
        return config.patterns.get(field_name, [])


def save_raw_text_to_vendor_folder(raw_text: str, pdf_path: Path, vendor_folder: Optional[str] = None, extraction_method: str = "text_extraction") -> Optional[Path]:
    """
    Save raw extracted text to vendor-specific raw_text folder in JSON format.

    Args:
        raw_text: The raw extracted text content
        pdf_path: Path to the original PDF file
        vendor_folder: Vendor-specific folder (e.g., 'ceco', 'steelcraft', 'schlage')
        extraction_method: Method used for extraction (e.g., 'text_extraction', 'ocr')

    Returns:
        Path to the saved raw text file, or None if vendor_folder not provided
    """
    if not vendor_folder:
        return None

    try:
        from datetime import datetime

        # Create vendor-specific raw_text directory
        raw_text_dir = Path("output") / vendor_folder / "raw_text"
        raw_text_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename based on PDF name
        base_name = pdf_path.stem
        raw_text_filename = f"{base_name}_raw_text.json"
        raw_text_path = raw_text_dir / raw_text_filename

        # Create structured raw text data
        raw_text_data = {
            "metadata": {
                "pdf_path": str(pdf_path),
                "pdf_filename": pdf_path.name,
                "vendor_folder": vendor_folder,
                "extraction_method": extraction_method,
                "extraction_timestamp": datetime.now().isoformat(),
                "processor": "TextPDFProcessor"
            },
            "raw_text": raw_text,
            "text_length": len(raw_text),
            "line_count": len(raw_text.split('\n')) if raw_text else 0
        }

        # Save raw text as JSON
        with open(raw_text_path, 'w', encoding='utf-8') as f:
            json.dump(raw_text_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved raw text JSON to: {raw_text_path}")
        return raw_text_path

    except Exception as e:
        logger.error(f"Failed to save raw text: {e}")
        return None


class DynamicPatternExtractor:
    """Handles dynamic pattern extraction with confidence scoring."""

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.pattern_cache = {}

    def extract_field_with_confidence(self, text: str, field_name: str, vendor_type: str = 'generic') -> ExtractionResult:
        """Extract a field with confidence scoring."""
        patterns = self.config_manager.get_patterns(vendor_type, field_name)
        if not patterns:
            return ExtractionResult(value=None, confidence=0.0, method="no_patterns")

        results = []

        for i, pattern_obj in enumerate(patterns):
            try:
                matches = re.findall(pattern_obj.pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    value = match if isinstance(match, str) else (match[0] if match else "")
                    if value and value.strip():
                        confidence = self._calculate_pattern_confidence(
                            pattern_obj, value, text, i, len(patterns)
                        )
                        results.append(ExtractionResult(
                            value=value.strip(),
                            confidence=confidence,
                            pattern_used=pattern_obj.pattern,
                            method="regex_pattern",
                            metadata={
                                "pattern_description": pattern_obj.description,
                                "pattern_index": i,
                                "vendor_specific": pattern_obj.vendor_specific
                            }
                        ))
            except re.error as e:
                logger.warning(f"Invalid regex pattern for {field_name}: {pattern_obj.pattern} - {e}")
                continue

        if not results:
            return ExtractionResult(value=None, confidence=0.0, method="no_matches")

        # Return the highest confidence result
        best_result = max(results, key=lambda x: x.confidence)
        return best_result

    def _calculate_pattern_confidence(self, pattern_obj: FieldPattern, match_value: str,
                                    full_text: str, pattern_index: int, total_patterns: int) -> float:
        """Calculate confidence score for a pattern match."""
        # Base confidence from pattern weight and position
        base_confidence = pattern_obj.confidence_weight * (1.0 - (pattern_index * 0.1))

        # Adjust based on match quality
        quality_score = self._assess_match_quality(match_value, pattern_obj)

        # Context validation
        context_score = self._validate_context(match_value, full_text, pattern_obj)

        # Combine scores
        final_confidence = min(base_confidence * quality_score * context_score, 1.0)
        return max(final_confidence, 0.0)

    def _assess_match_quality(self, match_value: str, pattern_obj: FieldPattern) -> float:
        """Assess the quality of a match."""
        if not match_value or not match_value.strip():
            return 0.0

        # Length-based assessment
        length_score = 1.0
        if len(match_value) < 2:
            length_score = 0.3
        elif len(match_value) > 50:
            length_score = 0.7

        # Content-based assessment
        content_score = 1.0
        if match_value.isdigit() and len(match_value) < 3:
            content_score = 0.5  # Short numbers are less reliable

        return length_score * content_score

    def _validate_context(self, match_value: str, full_text: str, pattern_obj: FieldPattern) -> float:
        """Validate match against context requirements."""
        if not pattern_obj.required_context:
            return 1.0

        # Find the position of the match in text
        match_pos = full_text.find(match_value)
        if match_pos == -1:
            return 0.8  # Slight penalty if exact match not found

        # Check surrounding context (100 characters before and after)
        start = max(0, match_pos - 100)
        end = min(len(full_text), match_pos + len(match_value) + 100)
        context = full_text[start:end].lower()

        # Check for required context keywords
        context_matches = sum(1 for keyword in pattern_obj.required_context
                            if keyword.lower() in context)

        if context_matches == 0:
            return 0.6  # Penalty for missing context
        elif context_matches >= len(pattern_obj.required_context):
            return 1.2  # Bonus for all context present
        else:
            return 0.8 + (context_matches / len(pattern_obj.required_context)) * 0.4


class TextPDFProcessor:
    """
    Enhanced text-selectable PDF processor with dynamic pattern extraction.
    """

    def __init__(self, vendor_type: str = 'generic'):
        """Initialize the text PDF processor."""
        self.extracted_elements = []
        self.page_layouts = []
        self.vendor_type = vendor_type
        self.config_manager = ConfigurationManager()
        self.pattern_extractor = DynamicPatternExtractor(self.config_manager)
        self.vendor_config = self.config_manager.get_config(vendor_type)
    
    def set_vendor_type(self, vendor_type: str) -> None:
        """Update the vendor type and reload configuration."""
        self.vendor_type = vendor_type
        self.vendor_config = self.config_manager.get_config(vendor_type)
        logger.info(f"Updated vendor type to: {vendor_type}")

    def process_pdf(self, pdf_path: str, vendor_folder: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a text-selectable PDF and extract structured data.

        Args:
            pdf_path: Path to the PDF file
            vendor_folder: Optional vendor-specific folder for saving raw text

        Returns:
            Structured data dictionary maintaining PDF layout
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path_obj}")

        logger.info(f"Processing text-selectable PDF: {pdf_path_obj.name}")

        # Extract text with layout information using pdfplumber
        layout_data = self._extract_with_layout(pdf_path_obj)

        # Save raw text to vendor-specific folder if vendor_folder is provided
        # Use exact raw text if available, otherwise fall back to processed text
        raw_text = layout_data.get('all_raw_text', '') or layout_data.get('all_text', '')
        if vendor_folder and raw_text:
            save_raw_text_to_vendor_folder(raw_text, pdf_path_obj, vendor_folder)

        # Extract structured data from the layout-aware text
        structured_data = self._extract_structured_data(layout_data)

        # Add enhanced metadata
        structured_data['metadata'] = {
            'pdf_path': str(pdf_path_obj),
            'extraction_method': 'enhanced_text_extraction',
            'total_pages': len(layout_data.get('pages', [])),
            'processor': 'EnhancedTextPDFProcessor',
            'vendor_type': self.vendor_type,
            'processing_timestamp': datetime.now().isoformat(),
            'confidence_score': structured_data.get('extraction_metadata', {}).get('overall_confidence', 0.0)
        }

        # Add validation results
        validation_results = self._validate_extraction_results(structured_data)
        structured_data['validation'] = validation_results

        logger.info(f"Enhanced text extraction complete for {pdf_path_obj.name} with {structured_data['metadata']['confidence_score']:.1%} confidence")
        return structured_data

    def _validate_extraction_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extraction results and provide quality metrics."""
        validation = {
            "has_invoice_number": bool(data.get("invoice_header", {}).get("invoice_number")),
            "has_invoice_date": bool(data.get("invoice_header", {}).get("invoice_date")),
            "has_customer_po": bool(data.get("invoice_header", {}).get("customer_po")),
            "has_line_items": len(data.get("line_items", [])) > 0,
            "has_vendor_info": bool(data.get("vendor_info", {}).get("company_name")),
            "has_customer_info": bool(data.get("customer_info", {}).get("company_name")),
            "has_totals": bool(data.get("totals", {})),
            "line_item_count": len(data.get("line_items", [])),
            "extraction_quality": self._assess_extraction_quality(data)
        }

        # Calculate completeness score
        required_fields = ["has_invoice_number", "has_invoice_date", "has_line_items", "has_vendor_info"]
        completeness = sum(validation[field] for field in required_fields) / len(required_fields)
        validation["completeness_score"] = completeness

        return validation

    def _assess_extraction_quality(self, data: Dict[str, Any]) -> str:
        """Assess overall extraction quality."""
        confidence = data.get('extraction_metadata', {}).get('overall_confidence', 0.0)

        if confidence >= 0.9:
            return "excellent"
        elif confidence >= 0.8:
            return "good"
        elif confidence >= 0.6:
            return "fair"
        elif confidence >= 0.4:
            return "poor"
        else:
            return "very_poor"
    
    def _extract_with_layout(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text while preserving layout information.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing text with layout information
        """
        layout_data = {
            'pages': [],
            'all_text': "",
            'all_raw_text': "",  # Add field for exact raw text
            'text_elements': []
        }

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_data = self._extract_page_layout(page, page_num + 1)
                    layout_data['pages'].append(page_data)
                    layout_data['all_text'] += page_data['text'] + "\n"
                    # Use raw text for exact extraction
                    if page_data.get('raw_text'):
                        layout_data['all_raw_text'] += f"=== PAGE {page_num + 1} ===\n"
                        layout_data['all_raw_text'] += page_data['raw_text'] + "\n"
                    layout_data['text_elements'].extend(page_data['elements'])

        except Exception as e:
            logger.error(f"Error extracting layout with pdfplumber: {e}")
            # Fallback to PyPDF2 for basic text extraction
            layout_data = self._fallback_pypdf2_extraction(pdf_path)

        return layout_data
    
    def _extract_page_layout(self, page, page_number: int) -> Dict[str, Any]:
        """
        Extract layout information from a single page.

        Args:
            page: pdfplumber page object
            page_number: Page number (1-based)

        Returns:
            Dictionary containing page layout data
        """
        page_data = {
            'page_number': page_number,
            'text': "",
            'raw_text': "",  # Add raw text field for exact extraction
            'elements': [],
            'tables': [],
            'lines': []
        }

        try:
            # Extract exact raw text preserving all spacing and layout
            page_data['raw_text'] = self._extract_raw_text_exact(page)

            # Extract text with character-level details for processing
            chars = page.chars
            if chars:
                # Group characters into text elements
                text_elements = self._group_characters_into_elements(chars, page_number)
                page_data['elements'] = text_elements
                page_data['text'] = " ".join([elem.text for elem in text_elements])
            else:
                # Fallback to simple text extraction
                page_data['text'] = page.extract_text() or ""

            # Extract tables if present
            tables = page.extract_tables()
            if tables:
                page_data['tables'] = self._process_tables(tables)

            # Extract lines/rules for layout understanding
            lines = page.lines
            if lines:
                page_data['lines'] = [
                    {
                        'x0': line['x0'], 'y0': line['y0'],
                        'x1': line['x1'], 'y1': line['y1']
                    }
                    for line in lines
                ]
                
        except Exception as e:
            logger.warning(f"Error extracting page {page_number} layout: {e}")
            # Fallback to basic text extraction
            page_data['text'] = page.extract_text() or ""
        
        return page_data
    
    def _group_characters_into_elements(self, chars: List[Dict], page_number: int) -> List[TextElement]:
        """
        Group individual characters into meaningful text elements.
        
        Args:
            chars: List of character dictionaries from pdfplumber
            page_number: Page number
            
        Returns:
            List of TextElement objects
        """
        if not chars:
            return []
        
        elements = []
        current_text = ""
        current_bbox = None
        current_font = None
        current_size = None
        
        for char in chars:
            char_text = char.get('text', '')
            char_bbox = (char.get('x0', 0), char.get('y0', 0), 
                        char.get('x1', 0), char.get('y1', 0))
            char_font = char.get('fontname', '')
            char_size = char.get('size', 0)
            
            # Check if this character continues the current element
            if (current_bbox and current_font == char_font and 
                current_size == char_size and 
                abs(char_bbox[1] - current_bbox[1]) < 2):  # Same line
                
                current_text += char_text
                current_bbox = (
                    min(current_bbox[0], char_bbox[0]),
                    min(current_bbox[1], char_bbox[1]),
                    max(current_bbox[2], char_bbox[2]),
                    max(current_bbox[3], char_bbox[3])
                )
            else:
                # Save previous element if it exists
                if current_text.strip() and current_bbox is not None:
                    elements.append(TextElement(
                        text=current_text.strip(),
                        x0=current_bbox[0] if current_bbox else 0, y0=current_bbox[1] if current_bbox else 0,
                        x1=current_bbox[2] if current_bbox else 0, y1=current_bbox[3] if current_bbox else 0,
                        fontname=current_font,
                        fontsize=current_size,
                        page_number=page_number
                    ))
                
                # Start new element
                current_text = char_text
                current_bbox = char_bbox
                current_font = char_font
                current_size = char_size
        
        # Add the last element
        if current_text.strip() and current_bbox is not None:
            elements.append(TextElement(
                text=current_text.strip(),
                x0=current_bbox[0] if current_bbox else 0, y0=current_bbox[1] if current_bbox else 0,
                x1=current_bbox[2] if current_bbox else 0, y1=current_bbox[3] if current_bbox else 0,
                fontname=current_font,
                fontsize=current_size,
                page_number=page_number
            ))
        
        return elements

    def _extract_raw_text_exact(self, page) -> str:
        """
        Extract exact raw text from page preserving all spacing and layout.

        Args:
            page: pdfplumber page object

        Returns:
            Raw text with exact spacing and formatting preserved
        """
        try:
            # Method 1: Try to get text with layout preservation using extract_text with layout=True
            try:
                raw_text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=1)
                if raw_text and len(raw_text.strip()) > 0:
                    return raw_text
            except Exception as e:
                logger.debug(f"Layout-based extraction failed: {e}")

            # Method 2: Character-by-character reconstruction preserving exact positions
            chars = page.chars
            if chars:
                return self._reconstruct_text_with_exact_spacing(chars)

            # Method 3: Fallback to basic extraction
            return page.extract_text() or ""

        except Exception as e:
            logger.warning(f"Raw text extraction failed: {e}")
            return ""

    def _reconstruct_text_with_exact_spacing(self, chars: List[Dict]) -> str:
        """
        Reconstruct text from characters preserving exact spacing and layout.

        Args:
            chars: List of character dictionaries from pdfplumber

        Returns:
            Text with exact spacing preserved
        """
        if not chars:
            return ""

        # Sort characters by y-coordinate (top to bottom) then x-coordinate (left to right)
        sorted_chars = sorted(chars, key=lambda c: (-c.get('y0', 0), c.get('x0', 0)))

        lines = []
        current_line = []
        current_y = None
        y_tolerance = 2  # Tolerance for considering characters on the same line

        for char in sorted_chars:
            char_y = char.get('y0', 0)
            char_text = char.get('text', '')
            char_x = char.get('x0', 0)

            # Check if this character is on a new line
            if current_y is None or abs(char_y - current_y) > y_tolerance:
                # Save previous line if it exists
                if current_line:
                    lines.append(self._build_line_with_spacing(current_line))

                # Start new line
                current_line = [(char_x, char_text)]
                current_y = char_y
            else:
                # Add to current line
                current_line.append((char_x, char_text))

        # Add the last line
        if current_line:
            lines.append(self._build_line_with_spacing(current_line))

        return '\n'.join(lines)

    def _build_line_with_spacing(self, char_positions: List[tuple]) -> str:
        """
        Build a line of text with proper spacing based on character positions.

        Args:
            char_positions: List of (x_position, character) tuples

        Returns:
            Line of text with proper spacing
        """
        if not char_positions:
            return ""

        # Sort by x position
        char_positions.sort(key=lambda x: x[0])

        line_text = ""
        prev_x = None

        for x_pos, char_text in char_positions:
            if prev_x is not None:
                # Calculate spacing based on position difference
                # Approximate character width (adjust as needed)
                char_width = 6  # Average character width in points
                space_count = max(0, int((x_pos - prev_x) / char_width) - 1)

                # Add spaces if there's a gap
                if space_count > 0:
                    line_text += ' ' * space_count

            line_text += char_text
            prev_x = x_pos + len(char_text) * 6  # Estimate end position

        return line_text

    def _process_tables(self, tables: List[List[List[str]]]) -> List[Dict[str, Any]]:
        """
        Process extracted tables into structured format.
        
        Args:
            tables: List of tables from pdfplumber
            
        Returns:
            List of processed table dictionaries
        """
        processed_tables = []
        
        for table_idx, table in enumerate(tables):
            if not table or not table[0]:
                continue
                
            # Assume first row contains headers
            headers = [cell.strip() if cell else f"Column_{i}" 
                      for i, cell in enumerate(table[0])]
            
            rows = []
            for row in table[1:]:
                if row and any(cell and cell.strip() for cell in row):
                    row_dict = {}
                    for i, cell in enumerate(row):
                        header = headers[i] if i < len(headers) else f"Column_{i}"
                        row_dict[header] = cell.strip() if cell else ""
                    rows.append(row_dict)
            
            processed_tables.append({
                'table_index': table_idx,
                'headers': headers,
                'rows': rows,
                'row_count': len(rows)
            })
        
        return processed_tables
    
    def _fallback_pypdf2_extraction(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Fallback text extraction using PyPDF2.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Basic text extraction data
        """
        layout_data = {
            'pages': [],
            'all_text': "",
            'text_elements': []
        }
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    page_data = {
                        'page_number': page_num + 1,
                        'text': page_text,
                        'elements': [],
                        'tables': [],
                        'lines': []
                    }
                    layout_data['pages'].append(page_data)
                    layout_data['all_text'] += page_text + "\n"
                    
        except Exception as e:
            logger.error(f"PyPDF2 fallback extraction failed: {e}")
        
        return layout_data

    def _extract_structured_data(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from layout-aware text using dynamic patterns with enhanced error handling.

        Args:
            layout_data: Dictionary containing text with layout information

        Returns:
            Structured data dictionary with confidence scores and error handling
        """
        all_text = layout_data.get('all_text', '')

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
                "confidence_scores": {},
                "extraction_methods": {},
                "vendor_type": self.vendor_type,
                "processing_timestamp": datetime.now().isoformat(),
                "errors": [],
                "warnings": []
            }
        }

        # Track extraction errors and warnings
        extraction_errors = []
        extraction_warnings = []

        # Extract invoice header information with error handling
        try:
            header_results = self._extract_invoice_header_enhanced(all_text)
            structured_data["invoice_header"] = header_results["data"]
            structured_data["extraction_metadata"]["confidence_scores"]["invoice_header"] = header_results["confidence"]
        except Exception as e:
            error_msg = f"Error extracting invoice header: {str(e)}"
            extraction_errors.append(error_msg)
            logger.warning(error_msg)
            structured_data["invoice_header"] = {}
            structured_data["extraction_metadata"]["confidence_scores"]["invoice_header"] = 0.0

        # Extract vendor information with error handling
        try:
            vendor_results = self._extract_vendor_info_enhanced(all_text)
            structured_data["vendor_info"] = vendor_results["data"]
            structured_data["extraction_metadata"]["confidence_scores"]["vendor_info"] = vendor_results["confidence"]
        except Exception as e:
            error_msg = f"Error extracting vendor info: {str(e)}"
            extraction_errors.append(error_msg)
            logger.warning(error_msg)
            structured_data["vendor_info"] = {}
            structured_data["extraction_metadata"]["confidence_scores"]["vendor_info"] = 0.0

        # Extract customer information with error handling
        try:
            customer_results = self._extract_customer_info_enhanced(all_text)
            structured_data["customer_info"] = customer_results["data"]
            structured_data["extraction_metadata"]["confidence_scores"]["customer_info"] = customer_results["confidence"]
        except Exception as e:
            error_msg = f"Error extracting customer info: {str(e)}"
            extraction_errors.append(error_msg)
            logger.warning(error_msg)
            structured_data["customer_info"] = {}
            structured_data["extraction_metadata"]["confidence_scores"]["customer_info"] = 0.0

        # Extract additional address information with error handling
        structured_data["sold_to"] = {}
        structured_data["ship_to"] = {}
        structured_data["remit_to"] = {}
        try:
            self._extract_address_info_enhanced(all_text, structured_data)
        except Exception as e:
            warning_msg = f"Warning extracting address info: {str(e)}"
            extraction_warnings.append(warning_msg)
            logger.warning(warning_msg)

        # Extract line items with enhanced processing and error handling
        try:
            line_items_results = self._extract_line_items_enhanced(layout_data)
            structured_data["line_items"] = line_items_results["data"]
            structured_data["extraction_metadata"]["confidence_scores"]["line_items"] = line_items_results["confidence"]

            # Check if line items extraction was successful
            if not structured_data["line_items"]:
                warning_msg = "No line items extracted - this may indicate a parsing issue"
                extraction_warnings.append(warning_msg)
                logger.warning(warning_msg)
        except Exception as e:
            error_msg = f"Error extracting line items: {str(e)}"
            extraction_errors.append(error_msg)
            logger.error(error_msg)
            structured_data["line_items"] = []
            structured_data["extraction_metadata"]["confidence_scores"]["line_items"] = 0.0

        # Extract totals with error handling
        try:
            totals_results = self._extract_totals_enhanced(all_text)
            structured_data["totals"] = totals_results["data"]
            structured_data["extraction_metadata"]["confidence_scores"]["totals"] = totals_results["confidence"]
        except Exception as e:
            error_msg = f"Error extracting totals: {str(e)}"
            extraction_errors.append(error_msg)
            logger.warning(error_msg)
            structured_data["totals"] = {}
            structured_data["extraction_metadata"]["confidence_scores"]["totals"] = 0.0

        # Extract payment terms with error handling
        try:
            payment_results = self._extract_payment_terms_enhanced(all_text)
            structured_data["payment_terms"] = payment_results["data"]
            structured_data["extraction_metadata"]["confidence_scores"]["payment_terms"] = payment_results["confidence"]
        except Exception as e:
            error_msg = f"Error extracting payment terms: {str(e)}"
            extraction_errors.append(error_msg)
            logger.warning(error_msg)
            structured_data["payment_terms"] = {}
            structured_data["extraction_metadata"]["confidence_scores"]["payment_terms"] = 0.0

        # Extract shipping information with error handling
        try:
            shipping_results = self._extract_shipping_info_enhanced(all_text)
            structured_data["shipping_info"] = shipping_results["data"]
            structured_data["extraction_metadata"]["confidence_scores"]["shipping_info"] = shipping_results["confidence"]
        except Exception as e:
            error_msg = f"Error extracting shipping info: {str(e)}"
            extraction_errors.append(error_msg)
            logger.warning(error_msg)
            structured_data["shipping_info"] = {}
            structured_data["extraction_metadata"]["confidence_scores"]["shipping_info"] = 0.0

        # Store errors and warnings
        structured_data["extraction_metadata"]["errors"] = extraction_errors
        structured_data["extraction_metadata"]["warnings"] = extraction_warnings

        # Calculate overall confidence score with error consideration
        confidence_scores = structured_data["extraction_metadata"]["confidence_scores"]
        overall_confidence = self._calculate_overall_confidence(confidence_scores)

        # Reduce confidence if there were errors
        if extraction_errors:
            error_penalty = min(len(extraction_errors) * 0.1, 0.3)  # Max 30% penalty
            overall_confidence = max(overall_confidence - error_penalty, 0.0)

        structured_data["extraction_metadata"]["overall_confidence"] = overall_confidence

        return structured_data

    def _calculate_overall_confidence(self, confidence_scores: Dict[str, float]) -> float:
        """Calculate overall confidence score from individual field confidences."""
        if not confidence_scores:
            return 0.0

        # Weight different sections by importance
        weights = {
            "invoice_header": 0.3,
            "line_items": 0.3,
            "vendor_info": 0.15,
            "customer_info": 0.15,
            "totals": 0.1
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for section, confidence in confidence_scores.items():
            weight = weights.get(section, 0.05)  # Default small weight for other sections
            weighted_sum += confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _extract_invoice_header_enhanced(self, text: str) -> Dict[str, Any]:
        """Extract invoice header with confidence scoring."""
        header_fields = ["invoice_number", "invoice_date", "order_number", "customer_po", "due_date"]
        extracted_data = {}
        field_confidences = {}

        for field in header_fields:
            result = self.pattern_extractor.extract_field_with_confidence(
                text, field, self.vendor_type
            )
            extracted_data[field] = result.value
            field_confidences[field] = result.confidence

        # Calculate section confidence
        section_confidence = sum(field_confidences.values()) / len(field_confidences) if field_confidences else 0.0

        return {
            "data": extracted_data,
            "confidence": section_confidence,
            "field_confidences": field_confidences
        }

    def _extract_vendor_info_enhanced(self, text: str) -> Dict[str, Any]:
        """Extract vendor information with confidence scoring."""
        vendor_fields = ["company_name", "address", "city_state_zip", "phone", "email"]
        extracted_data = {}
        field_confidences = {}

        for field in vendor_fields:
            result = self.pattern_extractor.extract_field_with_confidence(
                text, field, self.vendor_type
            )
            extracted_data[field] = result.value
            field_confidences[field] = result.confidence

        section_confidence = sum(field_confidences.values()) / len(field_confidences) if field_confidences else 0.0

        return {
            "data": extracted_data,
            "confidence": section_confidence,
            "field_confidences": field_confidences
        }

    def _extract_customer_info_enhanced(self, text: str) -> Dict[str, Any]:
        """Extract customer information with confidence scoring."""
        customer_fields = ["company_name", "address", "city_state_zip", "phone"]
        extracted_data = {}
        field_confidences = {}

        for field in customer_fields:
            result = self.pattern_extractor.extract_field_with_confidence(
                text, field, self.vendor_type
            )
            extracted_data[field] = result.value
            field_confidences[field] = result.confidence

        section_confidence = sum(field_confidences.values()) / len(field_confidences) if field_confidences else 0.0

        return {
            "data": extracted_data,
            "confidence": section_confidence,
            "field_confidences": field_confidences
        }

    def _extract_address_info_enhanced(self, text: str, structured_data: Dict[str, Any]) -> None:
        """Extract sold_to, ship_to, and remit_to information with enhanced patterns."""
        # For now, use the existing methods but with improved error handling
        try:
            self._extract_sold_to_info(text, structured_data["sold_to"])
            self._extract_ship_to_info(text, structured_data["ship_to"])
            self._extract_remit_to_info(text, structured_data["remit_to"])
        except Exception as e:
            logger.warning(f"Error extracting address information: {e}")

    def _extract_totals_enhanced(self, text: str) -> Dict[str, Any]:
        """Extract totals with confidence scoring."""
        total_fields = ["discount_amount", "total_sale", "tax", "invoice_total", "freight"]
        extracted_data = {}
        field_confidences = {}

        for field in total_fields:
            result = self.pattern_extractor.extract_field_with_confidence(
                text, field, self.vendor_type
            )
            extracted_data[field] = result.value
            field_confidences[field] = result.confidence

        section_confidence = sum(field_confidences.values()) / len(field_confidences) if field_confidences else 0.0

        return {
            "data": extracted_data,
            "confidence": section_confidence,
            "field_confidences": field_confidences
        }

    def _extract_payment_terms_enhanced(self, text: str) -> Dict[str, Any]:
        """Extract payment terms with confidence scoring."""
        payment_fields = ["terms", "due_date", "discount_date"]
        extracted_data = {}
        field_confidences = {}

        for field in payment_fields:
            result = self.pattern_extractor.extract_field_with_confidence(
                text, field, self.vendor_type
            )
            extracted_data[field] = result.value
            field_confidences[field] = result.confidence

        section_confidence = sum(field_confidences.values()) / len(field_confidences) if field_confidences else 0.0

        return {
            "data": extracted_data,
            "confidence": section_confidence,
            "field_confidences": field_confidences
        }

    def _extract_shipping_info_enhanced(self, text: str) -> Dict[str, Any]:
        """Extract shipping information with confidence scoring."""
        shipping_fields = ["tracking_number", "carrier", "ship_from", "shipping_method", "freight_charge"]
        extracted_data = {}
        field_confidences = {}

        for field in shipping_fields:
            result = self.pattern_extractor.extract_field_with_confidence(
                text, field, self.vendor_type
            )
            extracted_data[field] = result.value
            field_confidences[field] = result.confidence

        section_confidence = sum(field_confidences.values()) / len(field_confidences) if field_confidences else 0.0

        return {
            "data": extracted_data,
            "confidence": section_confidence,
            "field_confidences": field_confidences
        }

    def _extract_line_items_enhanced(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract line items with enhanced table detection and adaptive processing."""
        line_items = []
        extraction_metadata = {
            "methods_used": [],
            "table_detection_results": {},
            "confidence_scores": []
        }

        # Strategy 1: Enhanced table-based extraction
        table_results = self._extract_from_tables_adaptive(layout_data)
        if table_results["items"]:
            line_items.extend(table_results["items"])
            extraction_metadata["methods_used"].append("adaptive_table_extraction")
            extraction_metadata["table_detection_results"] = table_results["metadata"]

        # Strategy 2: Enhanced text-based extraction if no table results
        if not line_items:
            text_results = self._extract_from_text_adaptive(layout_data.get('all_text', ''))
            if text_results["items"]:
                line_items.extend(text_results["items"])
                extraction_metadata["methods_used"].append("adaptive_text_extraction")

        # Strategy 3: Hybrid approach - combine both methods
        if not line_items:
            hybrid_results = self._extract_hybrid_line_items(layout_data)
            line_items.extend(hybrid_results["items"])
            extraction_metadata["methods_used"].append("hybrid_extraction")

        # Enhance line items with additional data
        if line_items:
            self._enhance_line_items_comprehensive(line_items, layout_data)

        # Calculate confidence scores
        item_confidences = [self._calculate_line_item_confidence(item) for item in line_items]
        section_confidence = sum(item_confidences) / len(item_confidences) if item_confidences else 0.0

        return {
            "data": line_items,
            "confidence": section_confidence,
            "extraction_metadata": extraction_metadata,
            "item_confidences": item_confidences
        }

    def _extract_from_tables_adaptive(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract line items from tables with adaptive column mapping."""
        line_items = []
        detection_metadata = {
            "tables_found": 0,
            "line_item_tables": 0,
            "column_mappings": [],
            "confidence_scores": []
        }

        for page in layout_data.get('pages', []):
            for table_idx, table in enumerate(page.get('tables', [])):
                detection_metadata["tables_found"] += 1

                # Analyze table structure
                table_analysis = self._analyze_table_structure_enhanced(table)

                if table_analysis["is_line_item_table"]:
                    detection_metadata["line_item_tables"] += 1
                    detection_metadata["column_mappings"].append(table_analysis["column_mapping"])
                    detection_metadata["confidence_scores"].append(table_analysis["confidence"])

                    # Extract items from this table
                    table_items = self._extract_items_from_table(table, table_analysis)
                    line_items.extend(table_items)

        return {
            "items": line_items,
            "metadata": detection_metadata
        }

    def _analyze_table_structure_enhanced(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze table structure to determine if it contains line items."""
        headers = [str(h).lower() if h else "" for h in table.get('headers', [])]
        rows = table.get('rows', [])

        # Get field mappings from vendor config
        field_mappings = self.vendor_config.field_mappings

        # Analyze headers for line item indicators
        column_mapping = {}
        confidence_score = 0.0

        for field, possible_names in field_mappings.items():
            best_match_idx = -1
            best_match_score = 0.0

            for idx, header in enumerate(headers):
                for possible_name in possible_names:
                    if possible_name.lower() in header:
                        match_score = len(possible_name) / len(header) if header else 0
                        if match_score > best_match_score:
                            best_match_score = match_score
                            best_match_idx = idx

            if best_match_idx >= 0:
                column_mapping[field] = best_match_idx
                confidence_score += best_match_score

        # Normalize confidence score
        confidence_score = confidence_score / len(field_mappings) if field_mappings else 0.0

        # Check if this looks like a line item table
        required_fields = ["line_number", "description", "quantity", "unit_price"]
        found_required = sum(1 for field in required_fields if field in column_mapping)
        is_line_item_table = found_required >= 2 and confidence_score > 0.3

        return {
            "is_line_item_table": is_line_item_table,
            "confidence": confidence_score,
            "column_mapping": column_mapping,
            "headers": headers,
            "row_count": len(rows)
        }

    def _extract_items_from_table(self, table: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract line items from a table using the analyzed structure."""
        items = []
        column_mapping = analysis["column_mapping"]
        rows = table.get('rows', [])

        for row_idx, row in enumerate(rows):
            if not row or not any(cell and str(cell).strip() for cell in row):
                continue  # Skip empty rows

            item = self._parse_table_row_adaptive(row, column_mapping, row_idx)
            if item and self._validate_line_item_enhanced(item):
                items.append(item)

        return items

    def _parse_table_row_adaptive(self, row: List[Any], column_mapping: Dict[str, int], row_idx: int) -> Optional[Dict[str, Any]]:
        """Parse a table row using adaptive column mapping."""
        item = {
            "line_number": str(row_idx + 1),  # Default line number
            "plant": "",
            "item_code": "",
            "quantity_ordered": 0,
            "quantity_shipped": 0,
            "quantity_backordered": 0,
            "description": "",
            "list_price": 0.0,
            "discount_percent": 0.0,
            "unit_price": 0.0,
            "extended_amount": 0.0,
            "pricing_breakdown": {
                "base_pricing": {},
                "components": [],
                "subtotals": {},
                "taxes": {},
                "fees": {}
            },
            "product_details": {
                "specifications": "",
                "mark_numbers": [],
                "additional_info": []
            }
        }

        # Map columns to fields
        for field, col_idx in column_mapping.items():
            if col_idx < len(row) and row[col_idx] is not None:
                value = str(row[col_idx]).strip()
                if value:
                    if field in ["quantity_ordered", "quantity_shipped", "quantity_backordered"]:
                        item[field] = self._safe_int_convert(value)
                    elif field in ["list_price", "discount_percent", "unit_price", "extended_amount"]:
                        item[field] = self._safe_float_convert(value)
                    else:
                        item[field] = value

        # Calculate missing quantities if possible
        if item["quantity_ordered"] > 0 and item["quantity_shipped"] == 0:
            item["quantity_shipped"] = item["quantity_ordered"]

        if item["quantity_ordered"] > item["quantity_shipped"]:
            item["quantity_backordered"] = item["quantity_ordered"] - item["quantity_shipped"]

        return item if item.get("description") or item.get("item_code") else None

    def _extract_from_text_adaptive(self, text: str) -> Dict[str, Any]:
        """Extract line items from text using adaptive patterns."""
        items = []

        # Use multiple pattern strategies
        strategies = [
            self._extract_with_vendor_patterns,
            self._extract_with_generic_patterns,
            self._extract_with_flexible_patterns
        ]

        for strategy in strategies:
            strategy_items = strategy(text)
            if strategy_items:
                items.extend(strategy_items)
                break  # Use first successful strategy

        return {"items": items}

    def _extract_with_vendor_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract using vendor-specific patterns."""
        items = []

        vendor_type_str = str(self.vendor_type).lower()
        if vendor_type_str == 'ceco':
            # CECO-specific patterns
            patterns = [
                r"(\d{3})\s+(\d{3})\s+([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([^0-9]+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                r"(\d{3})\s+(\d{3})\s+([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+([^0-9]+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
            ]
        else:
            # Generic patterns
            patterns = [
                r"(\d{1,3})\s+([A-Z0-9]+)\s+(\d+)\s+([^0-9]+?)\s+([\d.]+)\s+([\d.]+)",
                r"(\d{1,3})\s+([^0-9]+?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)"
            ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                item = self._parse_pattern_match_adaptive(match, pattern)
                if item and self._validate_line_item_enhanced(item):
                    items.append(item)
            if items:
                break

        return items

    def _extract_with_generic_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract using generic patterns that work across vendors."""
        items = []
        lines = text.split('\n')

        for line in lines:
            if self._looks_like_line_item_enhanced(line):
                item = self._parse_line_adaptive(line)
                if item and self._validate_line_item_enhanced(item):
                    items.append(item)

        return items

    def _extract_with_flexible_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract using flexible patterns for unusual formats."""
        items = []

        # Look for any line with multiple numeric values that could be line items
        lines = text.split('\n')
        for i, line in enumerate(lines):
            numeric_count = len(re.findall(r'\d+\.?\d*', line))
            if numeric_count >= 3:  # Likely has quantities and prices
                item = self._parse_flexible_line(line, i + 1)
                if item and self._validate_line_item_enhanced(item):
                    items.append(item)

        return items

    def _extract_hybrid_line_items(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine table and text extraction for maximum coverage."""
        items = []

        # Try to find partial table data and supplement with text
        table_items = self._extract_from_tables_adaptive(layout_data)["items"]
        text_items = self._extract_from_text_adaptive(layout_data.get('all_text', ''))["items"]

        # Merge and deduplicate
        all_items = table_items + text_items
        unique_items = self._deduplicate_line_items(all_items)

        return {"items": unique_items}

    def _parse_pattern_match_adaptive(self, match: tuple, pattern: str) -> Optional[Dict[str, Any]]:
        """Parse a pattern match into a line item with adaptive field mapping."""
        if not match:
            return None

        # Determine field mapping based on match length and pattern
        if len(match) >= 11:  # Full CECO format with BO
            return {
                "line_number": match[0],
                "plant": match[1],
                "item_code": match[2],
                "quantity_ordered": self._safe_int_convert(match[3]),
                "quantity_shipped": self._safe_int_convert(match[4]),
                "quantity_backordered": self._safe_int_convert(match[5]),
                "description": match[6].strip(),
                "list_price": self._safe_float_convert(match[7]),
                "discount_percent": self._safe_float_convert(match[8]),
                "unit_price": self._safe_float_convert(match[9]),
                "extended_amount": self._safe_float_convert(match[10])
            }
        elif len(match) >= 10:  # Full format without BO
            return {
                "line_number": match[0],
                "plant": match[1] if len(match) > 6 else "",
                "item_code": match[2] if len(match) > 6 else match[1],
                "quantity_ordered": self._safe_int_convert(match[3] if len(match) > 6 else match[2]),
                "quantity_shipped": self._safe_int_convert(match[4] if len(match) > 6 else match[2]),
                "quantity_backordered": 0,
                "description": match[5] if len(match) > 6 else match[3],
                "list_price": self._safe_float_convert(match[6] if len(match) > 6 else match[4]),
                "discount_percent": self._safe_float_convert(match[7] if len(match) > 6 else ""),
                "unit_price": self._safe_float_convert(match[8] if len(match) > 6 else match[4]),
                "extended_amount": self._safe_float_convert(match[9] if len(match) > 6 else match[5])
            }
        else:
            return None

    def _looks_like_line_item_enhanced(self, line: str) -> bool:
        """Enhanced check if a line looks like it contains line item data."""
        if not line or len(line.strip()) < 10:
            return False

        # Count numeric patterns
        numeric_patterns = len(re.findall(r'\d+\.?\d*', line))

        # Look for line item indicators
        indicators = ['qty', 'price', 'amount', 'total', 'each', 'unit']
        has_indicators = any(indicator in line.lower() for indicator in indicators)

        # Check for structured format (starts with numbers)
        starts_with_numbers = bool(re.match(r'^\s*\d+', line))

        return numeric_patterns >= 2 and (has_indicators or starts_with_numbers)

    def _parse_line_adaptive(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a line adaptively to extract line item data."""
        parts = line.split()
        if len(parts) < 3:
            return None

        # Try to identify numeric and text parts
        numeric_parts = []
        text_parts = []

        for part in parts:
            if re.match(r'^\d+\.?\d*$', part):
                numeric_parts.append(self._safe_float_convert(part))
            else:
                text_parts.append(part)

        if len(numeric_parts) < 2:
            return None

        # Build item with available data
        return {
            "line_number": str(int(numeric_parts[0])) if numeric_parts[0] < 1000 else "1",
            "item_code": text_parts[0] if text_parts else "",
            "description": " ".join(text_parts[1:]) if len(text_parts) > 1 else " ".join(text_parts),
            "quantity_ordered": int(numeric_parts[1]) if len(numeric_parts) > 1 else 1,
            "quantity_shipped": int(numeric_parts[1]) if len(numeric_parts) > 1 else 1,
            "unit_price": numeric_parts[-2] if len(numeric_parts) >= 2 else 0.0,
            "extended_amount": numeric_parts[-1] if numeric_parts else 0.0
        }

    def _parse_flexible_line(self, line: str, line_num: int) -> Optional[Dict[str, Any]]:
        """Parse a line with flexible approach for unusual formats."""
        # Extract all numbers and text
        numbers = re.findall(r'\d+\.?\d*', line)
        text_parts = re.findall(r'[A-Za-z][A-Za-z0-9\s]*', line)

        if len(numbers) < 2:
            return None

        return {
            "line_number": str(line_num),
            "description": " ".join(text_parts[:3]) if text_parts else f"Item {line_num}",
            "quantity_ordered": self._safe_int_convert(numbers[0]) if numbers else 1,
            "unit_price": self._safe_float_convert(numbers[-2]) if len(numbers) >= 2 else 0.0,
            "extended_amount": self._safe_float_convert(numbers[-1]) if numbers else 0.0
        }

    def _validate_line_item_enhanced(self, item: Dict[str, Any]) -> bool:
        """Enhanced validation for line items with confidence thresholds."""
        if not item:
            return False

        # Check minimum required fields
        has_identifier = bool(item.get("line_number") or item.get("item_code"))
        has_description = bool(item.get("description", "").strip())
        has_quantity = item.get("quantity_ordered", 0) > 0 or item.get("quantity_shipped", 0) > 0
        has_pricing = item.get("unit_price", 0) > 0 or item.get("extended_amount", 0) > 0

        # Require at least 2 of these 4 criteria
        criteria_met = sum([has_identifier, has_description, has_quantity, has_pricing])

        return criteria_met >= 2

    def _calculate_line_item_confidence(self, item: Dict[str, Any]) -> float:
        """Calculate confidence score for a line item."""
        confidence = 0.0

        # Field completeness scoring
        if item.get("line_number"):
            confidence += 0.1
        if item.get("item_code"):
            confidence += 0.15
        if item.get("description", "").strip():
            confidence += 0.2
        if item.get("quantity_ordered", 0) > 0:
            confidence += 0.15
        if item.get("unit_price", 0) > 0:
            confidence += 0.2
        if item.get("extended_amount", 0) > 0:
            confidence += 0.2

        return min(confidence, 1.0)

    def _deduplicate_line_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate line items based on key fields."""
        seen = set()
        unique_items = []

        for item in items:
            # Create a key based on line number, item code, and description
            key = (
                item.get("line_number", ""),
                item.get("item_code", ""),
                item.get("description", "")[:50]  # First 50 chars of description
            )

            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        return unique_items

    def _enhance_line_items_comprehensive(self, line_items: List[Dict[str, Any]], layout_data: Dict[str, Any]) -> None:
        """Enhance line items with comprehensive additional data."""
        text = layout_data.get('all_text', '')

        for item in line_items:
            # Add default structures if missing
            if "pricing_breakdown" not in item:
                item["pricing_breakdown"] = {
                    "base_pricing": {},
                    "components": [],
                    "subtotals": {},
                    "taxes": {},
                    "fees": {}
                }

            if "product_details" not in item:
                item["product_details"] = {
                    "specifications": "",
                    "mark_numbers": [],
                    "additional_info": []
                }

            # Try to extract mark numbers and specifications
            line_number = item.get("line_number", "")
            if line_number:
                self._extract_product_specifications_enhanced(text, item, line_number)

    def _extract_product_specifications_enhanced(self, text: str, item: Dict[str, Any], line_number: str) -> None:
        """Enhanced product specification extraction."""
        try:
            # Use existing mark number extraction but with error handling
            lines = text.split('\n')
            mark_numbers = self._extract_mark_numbers_comprehensive(text, lines, line_number)
            item["product_details"]["mark_numbers"] = mark_numbers
        except Exception as e:
            logger.warning(f"Error extracting specifications for line {line_number}: {e}")
            item["product_details"]["mark_numbers"] = []

    def _extract_invoice_header(self, text: str, header_dict: Dict[str, Any]) -> None:
        """Extract invoice header information."""
        # CECO-specific patterns based on actual invoice structure
        patterns = {
            "invoice_number": [
                # CECO invoices have the invoice number at the beginning after a dot
                r"^\.\s*([0-9]{8})\s",  # Pattern: ". 01793690"
                r"^\.?\s*([0-9]{8})\s+\d{1,2}\/\d{1,2}\/\d{2,4}",  # Number followed by date
                # Fallback patterns
                r"invoice\s*number\s*:?\s*([A-Z0-9\-]{6,15})",
                r"invoice\s*#?\s*:?\s*([A-Z0-9\-]{6,15})",
                r"\b([0-9]{8})\b"  # 8-digit number
            ],
            "invoice_date": [
                # CECO invoices have date on second line after invoice number
                r"^\.\s*[0-9]{8}\s+(\d{1,2}\/\d{1,2}\/\d{2,4})",  # After invoice number
                r"invoice\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
                # Look for date patterns in the text
                r"\b(\d{1,2}\/\d{1,2}\/\d{2,4})\b"
            ],
            "order_number": [
                # CECO order numbers appear to be in format like "F3AL6A-01"
                r"\b([A-Z0-9]{6}-[0-9]{2})\b",  # Pattern: F3AL6A-01
                r"order\s*no\s*:?\s*([A-Z0-9\-]{3,20})",
                r"order\s*number\s*:?\s*([A-Z0-9\-]{3,20})"
            ],
            "customer_po": [
                # Customer PO can be various formats: "6605280", "370555-001", etc.
                r"\b([0-9]{6}-[0-9]{3})\b",  # Format: 370555-001
                r"(?<!01)\b([0-9]{7})\b(?!\s*\d{1,2}\/)",  # 7-digit not starting with 01 and not followed by date
                r"\b([0-9]{6,8})\b(?=\s+[A-Z][0-9A-Z]{5}-[0-9]{2})",  # Number before order number pattern
                r"customer\s*po\s*:?\s*([A-Z0-9\-]{3,20})",
                r"po\s*#?\s*:?\s*([A-Z0-9\-]{3,20})"
            ],
            "due_date": [
                # Due date appears in "TOTAL DUE IS PAYABLE ON" line
                r"total\s*due\s*is\s*payable\s*on\s*(\d{1,2}\/\d{1,2}\/\d{2,4})",
                r"payable\s*on\s*(\d{1,2}\/\d{1,2}\/\d{2,4})",
                r"due\s*date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})"
            ]
        }

        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    header_dict[field] = match.group(1)
                    break

    def _extract_vendor_info(self, text: str, vendor_dict: Dict[str, Any]) -> None:
        """Extract vendor information."""
        # More flexible patterns for different vendors
        patterns = {
            "company_name": [
                r"(Ceco\s+Door\s+Products)",  # Exact match for CECO
                r"(Ceco\s+Door[^\\n]*)",
                r"([A-Z][A-Za-z\s&]+(?:Inc|LLC|Corp|Company|Co\.|Ltd|Products|Services|Solutions)\.?)",
                r"([A-Z][A-Za-z\s&]+Door[A-Za-z\s]*)",
                r"([A-Z][A-Za-z\s&]+Manufacturing[A-Za-z\s]*)"
            ],
            "address": [
                r"(\d+\s+[A-Z\s]+DR\.?)",
                r"(\d+\s+[A-Z][A-Za-z\s]+(?:DR|DRIVE|ST|STREET|AVE|AVENUE|BLVD|BOULEVARD|RD|ROAD)\.?)",
                r"(\d+\s+[A-Z][A-Za-z\s]{5,30})"
            ],
            "city_state_zip": [
                r"([A-Z]+,\s+[A-Z]{2}\s+\d{5})",
                r"([A-Z][A-Za-z\s]+),?\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)",
                r"([A-Z][A-Za-z\s]+)\s+([A-Z]{2})\s+(\d{5})"
            ],
            "phone": [
                r"(\(\d{3}\)\s+\d{3}-\d{4})",
                r"(\d{3})-(\d{3})-(\d{4})",
                r"(\d{3})\.(\d{3})\.(\d{4})"
            ]
        }

        for field, pattern_list in patterns.items():
            if isinstance(pattern_list, str):
                pattern_list = [pattern_list]

            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    if field == "city_state_zip" and len(match.groups()) >= 3:
                        vendor_dict[field] = f"{match.group(1).strip()}, {match.group(2)} {match.group(3)}"
                    elif field == "phone" and len(match.groups()) >= 3:
                        vendor_dict[field] = f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
                    else:
                        value = match.group(1).strip()
                        if len(value) >= 3 and not value.lower() in ['invoice', 'total', 'amount']:
                            vendor_dict[field] = value
                    break

    def _extract_customer_info(self, text: str, customer_dict: Dict[str, Any]) -> None:
        """Extract customer information."""
        # More flexible patterns for different customers
        patterns = {
            "sold_to_id": [
                r"sold\s+to\s*:?\s*(\d+)",
                r"customer\s*id\s*:?\s*(\d+)",
                r"account\s*#?\s*:?\s*(\d+)"
            ],
            "ship_to_id": [
                r"ship\s+to\s*:?\s*(\d+)",
                r"shipping\s*id\s*:?\s*(\d+)"
            ],
            "company_name": [
                r"(COOK\s+&?\s*BOARDMAN\s+[A-Z]+)",  # COOK & BOARDMAN JACKSONVILLE
                r"(COOK\s+&?\s*BOARDMAN[^\\n]*)",
                r"bill\s*to\s*:?\s*([A-Z][A-Za-z\s&]{5,50})",
                r"customer\s*:?\s*([A-Z][A-Za-z\s&]{5,50})",
                r"sold\s*to\s*:?\s*([A-Z][A-Za-z\s&]{5,50})",
                r"([A-Z][A-Za-z\s&]+(?:INC|LLC|CORP|COMPANY|CO\.|LTD)\.?)"
            ],
            "address": [
                r"(\d+\s+IMESON\s+PARK\s+BLVD)",
                r"(\d+\s+WESTPOINT\s+BLVD)",
                r"(\d+\s+[A-Z][A-Za-z\s]+(?:BLVD|BOULEVARD|ST|STREET|AVE|AVENUE|DR|DRIVE|RD|ROAD)\.?)",
                r"(\d+\s+[A-Z][A-Za-z\s]{5,30})"
            ],
            "suite": [
                r"(STE\s+\d+)",
                r"(SUITE\s+\d+)",
                r"(UNIT\s+\d+)",
                r"(APT\s+\d+)"
            ],
            "city_state_zip": [
                r"(JACKSONVILLE)\s+(FL)\s+(\d{5})",  # Handle JACKSONVILLE FL 32218
                r"(WINSTON\s+SALEM)\s+(NC)\s+(\d{5})",
                r"([A-Z][A-Za-z\s]+?)\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)",
                r"([A-Z][A-Za-z\s]+)\s+([A-Z]{2})\s+(\d{5})"
            ]
        }

        for field, pattern_list in patterns.items():
            if isinstance(pattern_list, str):
                pattern_list = [pattern_list]

            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    if field == "city_state_zip" and len(match.groups()) >= 3:
                        customer_dict[field] = f"{match.group(1).strip()}, {match.group(2)} {match.group(3)}"
                    else:
                        value = match.group(1).strip()
                        if len(value) >= 3 and not value.lower() in ['invoice', 'total', 'amount']:
                            customer_dict[field] = value
                    break

    def _extract_sold_to_info(self, text: str, sold_to_dict: Dict[str, Any]) -> None:
        """Extract 'Sold to' information."""
        patterns = {
            "id": [
                r"sold\s+to\s*:?\s*(\d{8})",  # 18010812
                r"(\d{8})\s+COOK\s+&\s+BOARDMAN",  # ID before company name
            ],
            "company_name": [
                r"(COOK\s+&\s+BOARDMAN\s+JACKSONVILLE)",  # Full name first
                r"(COOK\s+&\s+BOARDMAN\s+[A-Z]+)",
                r"sold\s+to\s*:?\s*\d{8}\s+([A-Z\s&]+(?:INC|LLC|CORP|COMPANY|CO\.|LTD)\.?)",
                r"(\d{8})\s+(COOK\s+&\s+BOARDMAN[^\\n]*)",
            ],
            "address": [
                r"(\d+\s+IMESON\s+PARK\s+BLVD)",
                r"(\d+\s+WESTPOINT\s+BLVD)",
                r"(\d+\s+[A-Z][A-Za-z\s]+(?:BLVD|BOULEVARD|ST|STREET|AVE|AVENUE|DR|DRIVE|RD|ROAD)\.?)",
            ],
            "suite": [
                r"(STE\s+\d+)",
                r"(SUITE\s+\d+)",
            ],
            "city_state_zip": [
                r"(JACKSONVILLE)\s+(FL)\s+(\d{5})",
                r"(WINSTON\s+SALEM)\s+(NC)\s+(\d{5})",
                r"([A-Z][A-Za-z\s]+?)\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)",
            ]
        }

        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    if field == "city_state_zip" and len(match.groups()) >= 3:
                        sold_to_dict[field] = f"{match.group(1).strip()}, {match.group(2)} {match.group(3)}"
                    elif field == "company_name" and len(match.groups()) >= 2:
                        sold_to_dict[field] = match.group(2).strip()
                    else:
                        sold_to_dict[field] = match.group(1).strip()
                    break

    def _extract_ship_to_info(self, text: str, ship_to_dict: Dict[str, Any]) -> None:
        """Extract 'Ship to' information."""
        # Look for ship to section - it's usually the same as sold to but might have different ID
        patterns = {
            "id": [
                r"ship\s+to\s*:?\s*(\d{8})",  # 18010812
                r"COOK\s+&\s+BOARDMAN[^(]*\((\d{8})\)",  # ID in parentheses after company name
            ],
            "company_name": [
                r"(COOK\s+&\s+BOARDMAN\s+JACKSONVILLE)",  # Full name first
                r"(COOK\s+&\s+BOARDMAN\s+[A-Z]+)",
                r"ship\s+to\s*:?\s*\d{8}\s+([A-Z\s&]+(?:INC|LLC|CORP|COMPANY|CO\.|LTD)\.?)",
                r"ship\s+to\s*:?\s*([A-Z\s&]+)\s+\(\d{8}\)",
            ],
            "address": [
                r"(\d+\s+IMESON\s+PARK\s+BLVD)",  # Use same patterns as sold_to since they're often the same
                r"(\d+\s+WESTPOINT\s+BLVD)",
                r"(\d+\s+[A-Z][A-Za-z\s]+(?:BLVD|BOULEVARD|ST|STREET|AVE|AVENUE|DR|DRIVE|RD|ROAD)\.?)",
            ],
            "suite": [
                r"(STE\s+\d+)",
                r"(SUITE\s+\d+)",
            ],
            "city_state_zip": [
                r"(JACKSONVILLE)\s+(FL)\s+(\d{5})",
                r"(WINSTON\s+SALEM)\s+(NC)\s+(\d{5})",
                r"([A-Z][A-Za-z\s]+?)\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)",
            ]
        }

        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    if field == "city_state_zip" and len(match.groups()) >= 3:
                        ship_to_dict[field] = f"{match.group(1).strip()}, {match.group(2)} {match.group(3)}"
                    else:
                        ship_to_dict[field] = match.group(1).strip()
                    break

    def _extract_remit_to_info(self, text: str, remit_to_dict: Dict[str, Any]) -> None:
        """Extract 'Remit to' information."""
        patterns = {
            "company_name": [
                r"please\s+remit\s+to\s*:?\s*([A-Z][A-Za-z\s]+)",
                r"remit\s+to\s*:?\s*([A-Z][A-Za-z\s]+)",
                r"(Ceco\s+Door\s+Products)",
            ],
            "address": [
                r"(\d{4}\s+Solutions\s+Center)",
                r"(\d+\s+[A-Z][A-Za-z\s]+Center)",
            ],
            "city_state_zip": [
                r"(Chicago)\s+(Illinois)\s+(\d{5}-\d{4})",
                r"([A-Z][A-Za-z\s]+)\s+([A-Z][A-Za-z]+)\s+(\d{5}(?:-\d{4})?)",
            ]
        }

        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    if field == "city_state_zip" and len(match.groups()) >= 3:
                        remit_to_dict[field] = f"{match.group(1).strip()}, {match.group(2)} {match.group(3)}"
                    else:
                        remit_to_dict[field] = match.group(1).strip()
                    break

    def _extract_line_items(self, layout_data: Dict[str, Any], line_items: List[Dict[str, Any]]) -> None:
        """Extract line items from tables or text."""
        # First try to extract from tables
        for page in layout_data.get('pages', []):
            for table in page.get('tables', []):
                if self._is_line_item_table(table):
                    for row in table['rows']:
                        line_item = self._parse_line_item_row(row)
                        if line_item:
                            line_items.append(line_item)

        # If no tables found, try text-based extraction
        if not line_items:
            text = layout_data.get('all_text', '')
            self._extract_line_items_from_text(text, line_items)

        # Sort line items by line number to maintain proper order
        if line_items:
            self._sort_line_items_by_line_number(line_items)

    def _is_line_item_table(self, table: Dict[str, Any]) -> bool:
        """Check if a table contains line items."""
        headers = [h.lower() for h in table.get('headers', [])]
        line_item_keywords = ['item', 'description', 'qty', 'quantity', 'price', 'amount']
        return any(keyword in ' '.join(headers) for keyword in line_item_keywords)

    def _parse_line_item_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a table row into a line item with enhanced pricing breakdown structure."""
        # This is a simplified parser - would need to be customized based on actual table structure
        line_item = {}

        for key, value in row.items():
            key_lower = key.lower()
            if 'line' in key_lower or 'item' in key_lower:
                line_item['line_number'] = value
            elif 'description' in key_lower:
                line_item['description'] = value
            elif 'qty' in key_lower or 'quantity' in key_lower:
                try:
                    line_item['quantity_ordered'] = int(value) if value.isdigit() else value
                    line_item['quantity_shipped'] = line_item['quantity_ordered']  # Assume same as ordered
                    line_item['quantity_backordered'] = 0  # Default to 0
                except:
                    line_item['quantity_ordered'] = value
                    line_item['quantity_shipped'] = value
                    line_item['quantity_backordered'] = 0
            elif 'price' in key_lower:
                try:
                    line_item['unit_price'] = float(value.replace('$', '').replace(',', ''))
                except:
                    line_item['unit_price'] = value
            elif 'amount' in key_lower or 'total' in key_lower:
                try:
                    line_item['extended_amount'] = float(value.replace('$', '').replace(',', ''))
                except:
                    line_item['extended_amount'] = value
            elif 'discount' in key_lower:
                try:
                    line_item['discount_percent'] = float(value.replace('%', '').replace(',', ''))
                except:
                    line_item['discount_percent'] = value
            elif 'list' in key_lower and 'price' in key_lower:
                try:
                    line_item['list_price'] = float(value.replace('$', '').replace(',', ''))
                except:
                    line_item['list_price'] = value

        # Add enhanced pricing breakdown structure if we have basic line item data
        if line_item:
            # Set default values for missing fields
            line_item.setdefault('plant', '')
            line_item.setdefault('item_code', '')
            line_item.setdefault('quantity_backordered', 0)
            line_item.setdefault('list_price', 0.0)
            line_item.setdefault('discount_percent', 0.0)
            line_item.setdefault('unit_price', 0.0)
            line_item.setdefault('extended_amount', 0.0)

            # Add enhanced pricing breakdown structure
            line_item['pricing_breakdown'] = {
                "base_pricing": {
                    "list_price": line_item.get('list_price', 0.0),
                    "discount_percent": line_item.get('discount_percent', 0.0),
                    "unit_price": line_item.get('unit_price', 0.0),
                    "extended_amount": line_item.get('extended_amount', 0.0)
                },
                "components": [],
                "subtotals": {},
                "taxes": {},
                "fees": {}
            }

            # Add product details structure
            line_item['product_details'] = {
                "specifications": "",
                "mark_numbers": [],
                "additional_info": []
            }

        return line_item if line_item else None

    def _extract_line_items_from_text(self, text: str, line_items: List[Dict[str, Any]]) -> None:
        """Extract line items from text using enhanced regex patterns with BO detection."""

        # Enhanced patterns with BO field detection - try both patterns to catch all line items
        patterns = [
            # Pattern with BO field: LINE PLANT ITEM QTY_ORD QTY_SHIP QTY_BO DESCRIPTION... LIST DISCOUNT UNIT EXTENDED
            r"(\d{3})\s+(\d{3})\s+([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([A-Z0-9\s]+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",

            # Pattern without BO field: LINE PLANT ITEM QTY_ORD QTY_SHIP DESCRIPTION... LIST DISCOUNT UNIT EXTENDED
            r"(\d{3})\s+(\d{3})\s+([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+([A-Z0-9\s]+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        ]

        # Try all patterns and collect all unique line items
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)

            for match in matches:
                # Determine if this match has BO field based on pattern index
                has_bo_field = (i == 0)  # First pattern has BO field
                line_item = self._parse_line_item_match_with_bo(match, has_bo_field)
                if line_item and self._validate_line_item(line_item):
                    # Check for duplicates
                    if not self._is_duplicate_line_item(line_item, line_items):
                        line_items.append(line_item)

        # If no structured patterns worked, try fallback extraction
        if not line_items:
            self._extract_line_items_fallback(text, line_items)

        # After extracting basic line items, enhance them with pricing breakdown components
        if line_items:
            self._enhance_line_items_with_pricing_breakdown(text, line_items)

    def _parse_line_item_from_line_bo(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single line to extract line item with BO detection using regex."""
        try:
            # Use regex to parse the line with proper BO detection
            # Pattern: LINE PLANT ITEM QTY_ORD QTY_SHIP [QTY_BO] DESCRIPTION... LIST_PRICE DISCOUNT UNIT_PRICE EXTENDED

            # First, try pattern with BO field
            pattern_with_bo = r'^(\d{3})\s+(\d{3})\s+([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$'
            match_with_bo = re.match(pattern_with_bo, line.strip())

            if match_with_bo:
                line_number = match_with_bo.group(1)
                plant = match_with_bo.group(2)
                item_code = match_with_bo.group(3)
                qty_ordered = int(match_with_bo.group(4))
                qty_shipped = int(match_with_bo.group(5))
                potential_bo = int(match_with_bo.group(6))
                description = match_with_bo.group(7).strip()
                list_price = float(match_with_bo.group(8))
                discount_percent = float(match_with_bo.group(9))
                unit_price = float(match_with_bo.group(10))
                extended_amount = float(match_with_bo.group(11))

                # Validate BO logic: ordered should equal shipped + bo
                if qty_ordered == qty_shipped + potential_bo:
                    qty_bo = potential_bo
                else:
                    # The number is not BO, include it in description
                    description = str(potential_bo) + " " + description
                    qty_bo = qty_ordered - qty_shipped if qty_ordered > qty_shipped else 0
            else:
                # Try pattern without BO field
                pattern_without_bo = r'^(\d{3})\s+(\d{3})\s+([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$'
                match_without_bo = re.match(pattern_without_bo, line.strip())

                if not match_without_bo:
                    return None

                line_number = match_without_bo.group(1)
                plant = match_without_bo.group(2)
                item_code = match_without_bo.group(3)
                qty_ordered = int(match_without_bo.group(4))
                qty_shipped = int(match_without_bo.group(5))
                description = match_without_bo.group(6).strip()
                list_price = float(match_without_bo.group(7))
                discount_percent = float(match_without_bo.group(8))
                unit_price = float(match_without_bo.group(9))
                extended_amount = float(match_without_bo.group(10))

                # Calculate BO if needed
                qty_bo = qty_ordered - qty_shipped if qty_ordered > qty_shipped else 0

            # Create line item
            line_item = {
                "line_number": line_number,
                "plant": plant,
                "item_code": item_code,
                "quantity_ordered": qty_ordered,
                "quantity_shipped": qty_shipped,
                "quantity_backordered": qty_bo,
                "description": description,
                "list_price": list_price,
                "discount_percent": discount_percent,
                "unit_price": unit_price,
                "extended_amount": extended_amount,
                # Enhanced pricing breakdown structure
                "pricing_breakdown": {
                    "base_pricing": {
                        "list_price": list_price,
                        "discount_percent": discount_percent,
                        "unit_price": unit_price,
                        "extended_amount": extended_amount
                    },
                    "components": [],
                    "subtotals": {},
                    "taxes": {},
                    "fees": {}
                },
                "product_details": {
                    "specifications": "",
                    "mark_numbers": [],
                    "additional_info": []
                }
            }

            return line_item

        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing line item from line: {e}")
            return None

    def _extract_line_items_with_patterns(self, text: str, line_items: List[Dict[str, Any]]) -> None:
        """Fallback extraction using regex patterns."""
        patterns = [
            # CECO specific pattern - exact format with proper spacing
            r"(\d{3})\s+(\d{3})\s+([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+([A-Z0-9\s]+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            # More flexible CECO pattern
            r"(\d{3})\s+(\d{3})\s+([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            # Generic line item pattern
            r"(\d{1,3})\s+(\d{2,3})\s+([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+([^0-9]+?)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                line_item = self._parse_line_item_match(match)
                if line_item and self._validate_line_item(line_item):
                    if not self._is_duplicate_line_item(line_item, line_items):
                        line_items.append(line_item)
            if line_items:
                break

    def _parse_line_item_match(self, match: tuple) -> Dict[str, Any]:
        """Parse a regex match into a line item dictionary with enhanced pricing breakdown."""
        line_item = {}

        try:
            if len(match) >= 11:  # Full format with BO field (11 fields)
                line_item = {
                    "line_number": match[0].strip(),
                    "plant": match[1].strip(),
                    "item_code": match[2].strip(),
                    "quantity_ordered": self._safe_int_convert(match[3]),
                    "quantity_shipped": self._safe_int_convert(match[4]),
                    "quantity_backordered": self._safe_int_convert(match[5]),  # BO field
                    "description": match[6].strip(),
                    "list_price": self._safe_float_convert(match[7]),
                    "discount_percent": self._safe_float_convert(match[8]),
                    "unit_price": self._safe_float_convert(match[9]),
                    "extended_amount": self._safe_float_convert(match[10]),
                    # Enhanced pricing breakdown structure
                    "pricing_breakdown": {
                        "base_pricing": {
                            "list_price": self._safe_float_convert(match[7]),
                            "discount_percent": self._safe_float_convert(match[8]),
                            "unit_price": self._safe_float_convert(match[9]),
                            "extended_amount": self._safe_float_convert(match[10])
                        },
                        "components": [],
                        "subtotals": {},
                        "taxes": {},
                        "fees": {}
                    },
                    "product_details": {
                        "specifications": "",
                        "mark_numbers": [],
                        "additional_info": []
                    }
                }
            elif len(match) >= 10:  # Full format without BO field (10 fields)
                line_item = {
                    "line_number": match[0].strip(),
                    "plant": match[1].strip(),
                    "item_code": match[2].strip(),
                    "quantity_ordered": self._safe_int_convert(match[3]),
                    "quantity_shipped": self._safe_int_convert(match[4]),
                    "quantity_backordered": 0,  # Default to 0 if no BO field
                    "description": match[5].strip(),
                    "list_price": self._safe_float_convert(match[6]),
                    "discount_percent": self._safe_float_convert(match[7]),
                    "unit_price": self._safe_float_convert(match[8]),
                    "extended_amount": self._safe_float_convert(match[9]),
                    # Enhanced pricing breakdown structure
                    "pricing_breakdown": {
                        "base_pricing": {
                            "list_price": self._safe_float_convert(match[6]),
                            "discount_percent": self._safe_float_convert(match[7]),
                            "unit_price": self._safe_float_convert(match[8]),
                            "extended_amount": self._safe_float_convert(match[9])
                        },
                        "components": [],
                        "subtotals": {},
                        "taxes": {},
                        "fees": {}
                    },
                    "product_details": {
                        "specifications": "",
                        "mark_numbers": [],
                        "additional_info": []
                    }
                }
            elif len(match) >= 6:  # Simplified format
                line_item = {
                    "line_number": match[0].strip(),
                    "item_code": match[1].strip(),
                    "quantity_ordered": self._safe_int_convert(match[2]),
                    "quantity_shipped": self._safe_int_convert(match[2]),  # Assume same as ordered
                    "quantity_backordered": 0,  # Default to 0
                    "description": match[3].strip(),
                    "unit_price": self._safe_float_convert(match[4]),
                    "extended_amount": self._safe_float_convert(match[5]),
                    "plant": "",
                    "list_price": 0.0,
                    "discount_percent": 0.0,
                    # Enhanced pricing breakdown structure
                    "pricing_breakdown": {
                        "base_pricing": {
                            "list_price": 0.0,
                            "discount_percent": 0.0,
                            "unit_price": self._safe_float_convert(match[4]),
                            "extended_amount": self._safe_float_convert(match[5])
                        },
                        "components": [],
                        "subtotals": {},
                        "taxes": {},
                        "fees": {}
                    },
                    "product_details": {
                        "specifications": "",
                        "mark_numbers": [],
                        "additional_info": []
                    }
                }
        except (IndexError, ValueError) as e:
            logger.warning(f"Error parsing line item match: {e}")
            return {}

        return line_item

    def _parse_line_item_match_with_bo(self, match: tuple, has_bo_field: bool) -> Dict[str, Any]:
        """Parse a regex match with BO field awareness."""
        line_item = {}

        try:
            if has_bo_field and len(match) >= 11:  # Pattern with BO field
                line_number = match[0].strip()
                plant = match[1].strip()
                item_code = match[2].strip()
                qty_ordered = self._safe_int_convert(match[3])
                qty_shipped = self._safe_int_convert(match[4])
                potential_bo = self._safe_int_convert(match[5])
                description = match[6].strip()
                list_price = self._safe_float_convert(match[7])
                discount_percent = self._safe_float_convert(match[8])
                unit_price = self._safe_float_convert(match[9])
                extended_amount = self._safe_float_convert(match[10])

                # Validate BO logic: ordered should equal shipped + bo
                if qty_ordered == qty_shipped + potential_bo:
                    qty_bo = potential_bo
                elif qty_ordered > qty_shipped and potential_bo == (qty_ordered - qty_shipped):
                    # Alternative BO validation: BO equals the difference
                    qty_bo = potential_bo
                else:
                    # The number is not BO, include it in description
                    description = str(potential_bo) + " " + description
                    qty_bo = qty_ordered - qty_shipped if qty_ordered > qty_shipped else 0

                line_item = {
                    "line_number": line_number,
                    "plant": plant,
                    "item_code": item_code,
                    "quantity_ordered": qty_ordered,
                    "quantity_shipped": qty_shipped,
                    "quantity_backordered": qty_bo,
                    "description": description,
                    "list_price": list_price,
                    "discount_percent": discount_percent,
                    "unit_price": unit_price,
                    "extended_amount": extended_amount,
                    # Enhanced pricing breakdown structure
                    "pricing_breakdown": {
                        "base_pricing": {
                            "list_price": list_price,
                            "discount_percent": discount_percent,
                            "unit_price": unit_price,
                            "extended_amount": extended_amount
                        },
                        "components": [],
                        "subtotals": {},
                        "taxes": {},
                        "fees": {}
                    },
                    "product_details": {
                        "specifications": "",
                        "mark_numbers": [],
                        "additional_info": []
                    }
                }
            elif len(match) >= 10:  # Pattern without BO field
                line_item = {
                    "line_number": match[0].strip(),
                    "plant": match[1].strip(),
                    "item_code": match[2].strip(),
                    "quantity_ordered": self._safe_int_convert(match[3]),
                    "quantity_shipped": self._safe_int_convert(match[4]),
                    "quantity_backordered": 0,  # Default to 0 if no BO field
                    "description": match[5].strip(),
                    "list_price": self._safe_float_convert(match[6]),
                    "discount_percent": self._safe_float_convert(match[7]),
                    "unit_price": self._safe_float_convert(match[8]),
                    "extended_amount": self._safe_float_convert(match[9]),
                    # Enhanced pricing breakdown structure
                    "pricing_breakdown": {
                        "base_pricing": {
                            "list_price": self._safe_float_convert(match[6]),
                            "discount_percent": self._safe_float_convert(match[7]),
                            "unit_price": self._safe_float_convert(match[8]),
                            "extended_amount": self._safe_float_convert(match[9])
                        },
                        "components": [],
                        "subtotals": {},
                        "taxes": {},
                        "fees": {}
                    },
                    "product_details": {
                        "specifications": "",
                        "mark_numbers": [],
                        "additional_info": []
                    }
                }

                # Calculate BO if ordered > shipped
                if line_item["quantity_ordered"] > line_item["quantity_shipped"]:
                    line_item["quantity_backordered"] = line_item["quantity_ordered"] - line_item["quantity_shipped"]

        except (IndexError, ValueError) as e:
            logger.warning(f"Error parsing line item match with BO: {e}")
            return {}

        return line_item

    def _parse_line_item_match_with_bo_detection(self, match: tuple, full_text: str) -> Dict[str, Any]:
        """Parse a regex match with smart BO field detection."""
        if len(match) < 10:
            return self._parse_line_item_match(match)

        # Extract the line number to find the specific line in the text
        line_number = match[0].strip()

        # Find the actual line in the text
        lines = full_text.split('\n')
        target_line = None
        for line in lines:
            if line.strip().startswith(line_number + " "):
                target_line = line.strip()
                break

        if not target_line:
            return self._parse_line_item_match(match)

        # Split the line into parts and analyze the structure
        parts = target_line.split()

        # Look for the pattern: line plant item qty_ord qty_shp [qty_bo] description... prices
        # If we have exactly the right number of numeric fields, we can detect BO
        try:
            # Find numeric sequences after the item code
            numeric_parts = []
            description_start_idx = -1

            for i, part in enumerate(parts):
                if i >= 3:  # Skip line, plant, item
                    if part.replace('.', '').isdigit():
                        numeric_parts.append((i, int(float(part))))
                    elif len(numeric_parts) >= 2:  # We've found at least qty_ord and qty_shp
                        description_start_idx = i
                        break

            # If we have 3 consecutive numeric values after item code, likely has BO
            if len(numeric_parts) >= 3 and description_start_idx > 0:
                # This line has BO field
                qty_ordered = numeric_parts[0][1]
                qty_shipped = numeric_parts[1][1]
                qty_bo = numeric_parts[2][1]

                # Calculate BO if it makes sense (ordered = shipped + bo)
                if qty_ordered == qty_shipped + qty_bo:
                    # Create enhanced match tuple with BO
                    enhanced_match = list(match)
                    if len(enhanced_match) == 10:
                        # Insert BO field: line, plant, item, qty_ord, qty_shp, qty_bo, desc, list, disc, unit, ext
                        enhanced_match = [
                            enhanced_match[0],  # line
                            enhanced_match[1],  # plant
                            enhanced_match[2],  # item
                            enhanced_match[3],  # qty_ord
                            enhanced_match[4],  # qty_shp
                            str(qty_bo),        # qty_bo (inserted)
                            enhanced_match[5],  # description
                            enhanced_match[6],  # list_price
                            enhanced_match[7],  # discount
                            enhanced_match[8],  # unit_price
                            enhanced_match[9]   # extended
                        ]
                        return self._parse_line_item_match(tuple(enhanced_match))
        except (ValueError, IndexError):
            pass

        # Fallback to regular parsing
        return self._parse_line_item_match(match)

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

    def _validate_line_item(self, line_item: Dict[str, Any]) -> bool:
        """Validate that a line item has required fields."""
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

    def _is_duplicate_line_item(self, new_item: Dict[str, Any], existing_items: List[Dict[str, Any]]) -> bool:
        """Check if line item is a duplicate."""
        for existing in existing_items:
            if (existing.get('line_number') == new_item.get('line_number') and
                existing.get('item_code') == new_item.get('item_code')):
                return True
        return False

    def _sort_line_items_by_line_number(self, line_items: List[Dict[str, Any]]) -> None:
        """Sort line items by their line number to maintain proper order."""
        def get_line_number_for_sorting(item: Dict[str, Any]) -> int:
            """Extract numeric line number for sorting."""
            line_number = item.get('line_number', '0')
            try:
                # Remove any non-numeric characters and convert to int
                numeric_part = ''.join(filter(str.isdigit, str(line_number)))
                return int(numeric_part) if numeric_part else 0
            except (ValueError, TypeError):
                return 0

        # Sort line items by line number
        line_items.sort(key=get_line_number_for_sorting)

    def _enhance_line_items_with_pricing_breakdown(self, text: str, line_items: List[Dict[str, Any]]) -> None:
        """Enhance line items with detailed pricing breakdown components."""
        lines = text.split('\n')

        for i, line_item in enumerate(line_items):
            line_number = line_item.get('line_number', '')

            # Find the line item in the text and extract associated pricing components
            self._extract_pricing_components_for_line_item(lines, line_item, line_number)
            self._extract_product_specifications_for_line_item(lines, line_item, line_number)

    def _extract_pricing_components_for_line_item(self, lines: List[str], line_item: Dict[str, Any], line_number: str) -> None:
        """Extract pricing components (MATERIAL AMOUNT, WELD AMOUNT, etc.) for a specific line item."""
        components = []

        # Look for the main line item first in the combined text
        all_text = " ".join(lines)

        # Find the line item section in the text
        line_item_pattern = rf"{line_number}\s+\d{{3}}\s+[A-Z0-9]+.*?(?=\d{{3}}\s+\d{{3}}\s+|$)"
        line_item_match = re.search(line_item_pattern, all_text, re.DOTALL)

        if not line_item_match:
            # Fallback: look line by line
            self._extract_pricing_components_line_by_line(lines, line_item, line_number)
            return

        line_item_text = line_item_match.group(0)

        # Look for pricing components within this line item section
        # Components typically appear as: "MATERIAL AMOUNT                           828.999  56.000     364.76     364.76"
        component_patterns = [
            r"(MATERIAL AMOUNT)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            r"(WELD AMOUNT)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            r"(LABOR AMOUNT)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            r"(FREIGHT AMOUNT)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            r"(TAX AMOUNT)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            r"([A-Z\s]+AMOUNT)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        ]

        # Check for component patterns in the line item text
        seen_components = set()
        for pattern in component_patterns:
            matches = re.findall(pattern, line_item_text)
            for match in matches:
                # Create a unique key for deduplication
                component_key = (match[0].strip(), match[1], match[2], match[3], match[4])
                if component_key not in seen_components:
                    seen_components.add(component_key)
                    component = {
                        "component_type": match[0].strip(),
                        "list_price": self._safe_float_convert(match[1]),
                        "discount_percent": self._safe_float_convert(match[2]),
                        "unit_price": self._safe_float_convert(match[3]),
                        "extended_amount": self._safe_float_convert(match[4])
                    }
                    components.append(component)

        # Update the line item with the extracted components
        if components:
            line_item["pricing_breakdown"]["components"] = components

            # Calculate totals for different component types
            material_total = sum(c["extended_amount"] for c in components if "MATERIAL" in c["component_type"])
            weld_total = sum(c["extended_amount"] for c in components if "WELD" in c["component_type"])
            labor_total = sum(c["extended_amount"] for c in components if "LABOR" in c["component_type"])

            line_item["pricing_breakdown"]["subtotals"] = {
                "material_amount": material_total,
                "weld_amount": weld_total,
                "labor_amount": labor_total,
                "total_components": sum(c["extended_amount"] for c in components)
            }

    def _extract_pricing_components_line_by_line(self, lines: List[str], line_item: Dict[str, Any], line_number: str) -> None:
        """Fallback method to extract pricing components line by line."""
        components = []

        # Look for the main line item first
        main_line_found = False
        main_line_index = -1

        for i, line in enumerate(lines):
            line = line.strip()
            # Check if this line contains our line number at the start
            if line.startswith(line_number + " ") and line_number:
                main_line_found = True
                main_line_index = i
                break

        if not main_line_found:
            return

        # Look for pricing components in the lines following the main line item
        component_patterns = [
            r"^(MATERIAL AMOUNT)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            r"^(WELD AMOUNT)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            r"^(LABOR AMOUNT)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            r"^(FREIGHT AMOUNT)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            r"^(TAX AMOUNT)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            r"^([A-Z\s]+AMOUNT)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        ]

        # Check the next few lines after the main line item for components
        for j in range(main_line_index + 1, min(main_line_index + 10, len(lines))):
            line = lines[j].strip()

            # Stop if we hit another line item or separator
            if re.match(r'^\d{3}\s+\d{3}\s+', line) or line.startswith('---'):
                break

            # Check for component patterns
            for pattern in component_patterns:
                match = re.match(pattern, line)
                if match:
                    component = {
                        "component_type": match.group(1).strip(),
                        "list_price": self._safe_float_convert(match.group(2)),
                        "discount_percent": self._safe_float_convert(match.group(3)),
                        "unit_price": self._safe_float_convert(match.group(4)),
                        "extended_amount": self._safe_float_convert(match.group(5))
                    }
                    components.append(component)
                    break

        # Update the line item with the extracted components
        if components:
            line_item["pricing_breakdown"]["components"] = components

            # Calculate totals for different component types
            material_total = sum(c["extended_amount"] for c in components if "MATERIAL" in c["component_type"])
            weld_total = sum(c["extended_amount"] for c in components if "WELD" in c["component_type"])
            labor_total = sum(c["extended_amount"] for c in components if "LABOR" in c["component_type"])

            line_item["pricing_breakdown"]["subtotals"] = {
                "material_amount": material_total,
                "weld_amount": weld_total,
                "labor_amount": labor_total,
                "total_components": sum(c["extended_amount"] for c in components)
            }

    def _extract_product_specifications_for_line_item(self, lines: List[str], line_item: Dict[str, Any], line_number: str) -> None:
        """Extract product specifications and mark numbers for a specific line item."""
        # Look for the line item section in the combined text
        all_text = " ".join(lines)

        # Find the line item section in the text - use a more comprehensive pattern
        # that captures everything until the line separator or next line item
        line_item_pattern = rf"{line_number}\s+\d{{3}}\s+[A-Z0-9]+.*?(?=\n-{{10,}}|\n\d{{3}}\s+\d{{3}}\s+[A-Z0-9]+|FREIGHT\s+CHARGE|SHIPMENT\s+TRACKING|$)"
        line_item_match = re.search(line_item_pattern, all_text, re.DOTALL)

        if not line_item_match:
            return

        line_item_text = line_item_match.group(0)

        # Extract only the mark numbers that belong to this specific line item
        line_item_specific_text = self._extract_line_item_specific_text(lines, line_number)

        specifications = []
        mark_numbers = []
        additional_info = []

        # Enhanced mark number extraction with multiple strategies
        mark_numbers = self._extract_mark_numbers_comprehensive(line_item_specific_text, lines, line_number)

        # Look for specifications (lines with product codes and specifications)
        spec_matches = re.findall(r'([A-Z]{2}\s+\d+[A-Z]+\d+[^0-9]*)', line_item_text)
        for spec_match in spec_matches:
            # Clean up the specification text
            spec_clean = re.sub(r'\s+', ' ', spec_match.strip())
            if spec_clean and len(spec_clean) > 5:
                specifications.append(spec_clean)

        # Look for additional product information patterns
        # Extract patterns like "70 RH FV3 H1 EAP A03 MS645H"
        additional_patterns = [
            r'(\d+\s+[LR]H\s+[A-Z0-9]+\s+H\d+\s+[A-Z0-9\s]+)',  # More specific pattern
            r'([A-Z]{2,3}\d+[A-Z])'  # Product codes like MS645H
        ]

        seen_additional = set()
        for pattern in additional_patterns:
            matches = re.findall(pattern, line_item_text)
            for match in matches:
                clean_match = re.sub(r'\s+', ' ', match.strip())
                # Filter out noise and duplicates
                if (clean_match and len(clean_match) > 3 and
                    clean_match not in seen_additional and
                    not re.match(r'^\d+\s+\d+\s+\d+', clean_match) and  # Skip numeric sequences
                    'AMOUNT' not in clean_match and  # Skip amount lines
                    'FRAME' not in clean_match):  # Skip frame descriptions
                    seen_additional.add(clean_match)
                    additional_info.append(clean_match)

        # Update the line item with extracted product details
        line_item["product_details"]["specifications"] = " | ".join(specifications) if specifications else ""
        line_item["product_details"]["mark_numbers"] = mark_numbers
        line_item["product_details"]["additional_info"] = additional_info

    def _extract_line_item_specific_text(self, lines: List[str], line_number: str) -> str:
        """
        Extract text that belongs specifically to this line item.

        This method finds the line item and extracts text from the line item start
        until the next line item, including any mark numbers that appear after page breaks.
        """
        # Join all lines into a single text for processing
        all_text = " ".join(lines)

        # Find the current line item position
        line_item_pattern = rf'{line_number}\s+\d{{3}}\s+[A-Z0-9]+'
        current_match = re.search(line_item_pattern, all_text)

        if not current_match:
            return ""

        start_pos = current_match.start()

        # Find the next line item to determine the end boundary
        # Look for the next line item header (more specific pattern)
        next_line_pattern = r'\d{3}\s+572\s+[A-Z0-9]+'  # More specific: includes plant code 572
        remaining_text = all_text[start_pos + len(current_match.group(0)):]

        next_match = re.search(next_line_pattern, remaining_text)
        if next_match:
            # Found next line item, extract text up to that point
            end_pos = start_pos + len(current_match.group(0)) + next_match.start()
            line_item_text = all_text[start_pos:end_pos]
        else:
            # No next line item found, extract to end but stop at shipping/payment sections
            line_item_text = all_text[start_pos:]

            # For the last line item, stop at shipping/payment sections
            end_patterns = [
                r'SHIPMENT\s+TRACKING',
                r'FREIGHT\s+CHARGE',
                r'TOTAL\s+DUE',
                r'Invoice\s+Number:',
                r'Please\s+Remit\s+to:'
            ]

            for pattern in end_patterns:
                match = re.search(pattern, line_item_text)
                if match:
                    line_item_text = line_item_text[:match.start()].strip()
                    break

        return line_item_text.strip()

    def _extract_mark_numbers_comprehensive(self, line_item_text: str, all_lines: List[str], line_number: str) -> List[str]:
        """
        Adaptive mark number extraction that learns from document patterns.

        This method uses multiple strategies to extract mark numbers regardless of format:
        1. Explicit label detection (MARK NO:, MARK #:, etc.)
        2. Pattern-based detection (alphanumeric codes, numbers with parentheses)
        3. Positional analysis (learning from document structure)
        4. Context-aware validation (avoiding false positives)
        """
        mark_numbers = []

        # Strategy 1: Explicit label-based extraction (flexible patterns)
        explicit_marks = self._extract_marks_with_labels(line_item_text)
        mark_numbers.extend(explicit_marks)

        # Strategy 2: Pattern-based extraction (no labels required)
        pattern_marks = self._extract_marks_by_patterns(line_item_text)
        mark_numbers.extend(pattern_marks)

        # Strategy 3: Positional/structural analysis
        positional_marks = self._extract_marks_by_position(all_lines, line_number)
        mark_numbers.extend(positional_marks)

        # Strategy 4: Context-aware validation and cleanup
        validated_marks = self._validate_and_clean_marks(mark_numbers, line_item_text)

        return validated_marks

    def _extract_marks_with_labels(self, text: str) -> List[str]:
        """Extract mark numbers with explicit labels using precise patterns."""
        mark_numbers = []

        # Precise pattern for explicit "MARK NO:" labels
        # This pattern captures mark numbers including comma-separated values
        label_pattern = r'MARK\s*NO\s*:\s+([A-Z0-9.,()]+(?:,[A-Z0-9.,()]+)*)'

        matches = re.findall(label_pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            # Clean and parse the match
            clean_match = match.strip()
            if clean_match:
                # Handle comma-separated values
                if ',' in clean_match:
                    marks = [m.strip() for m in clean_match.split(',') if m.strip()]
                    for mark in marks:
                        if self._is_valid_mark_number_format(mark):
                            mark_numbers.append(mark)
                else:
                    if self._is_valid_mark_number_format(clean_match):
                        mark_numbers.append(clean_match)

        return mark_numbers

    def _is_valid_mark_number_format(self, mark: str) -> bool:
        """Check if a string matches valid mark number formats."""
        if not mark or len(mark) < 1:
            return False

        # Exclude obvious prices (decimal numbers with 2 decimal places that look like currency)
        if re.match(r'^\d+\.\d{2}$', mark):
            # This looks like a price (e.g., 316.50, 123.45)
            return False

        # Valid mark number patterns based on actual invoice data:
        # - Decimal numbers: 101.2, 103.1, 167.1 (but not prices like 316.50)
        # - Alphanumeric with parentheses: HAP1(56), D001(56)
        # - Simple alphanumeric: HAP2, D001, HAP1
        # - Long numbers with parentheses: 8241799(40), 6797443(20), 7721190(20)
        # - Medium numbers with parentheses: 12345(20)

        valid_patterns = [
            r'^\d+\.\d{1}$',  # Decimal numbers like 101.2, 103.1 (single decimal place)
            r'^[A-Z]+\d+\(\d+\)$',  # HAP1(56), D001(56)
            r'^[A-Z]+\d+$',  # HAP2, D001, HAP1
            r'^\d{4,}\(\d+\)$',  # 8241799(40), 6797443(20), 7721190(20), 12345(20)
            r'^\d{6,}$',  # Long numbers like 8241799 (without parentheses)
        ]

        for pattern in valid_patterns:
            if re.match(pattern, mark, re.IGNORECASE):
                return True

        return False

    def _extract_marks_by_patterns(self, text: str) -> List[str]:
        """Extract mark numbers using precise patterns (no labels required)."""
        mark_numbers = []

        # Enhanced pattern-based extraction for cases without explicit labels
        # Focus on high-confidence patterns that are very likely to be mark numbers

        # High-confidence patterns based on actual invoice data:
        patterns = [
            # Long numbers with parentheses (very high confidence): 8241799(40), 7721190(20), 6797443(20)
            # Use lookahead/lookbehind to ensure proper boundaries without word boundary issues
            r'(?<!\d)(\d{7,8}\(\d{1,3}\))(?!\d)',

            # Decimal numbers that appear standalone: 101.2, 103.1
            r'(?<!\d)(\d{2,3}\.\d{1,2})(?!\d)',

            # Medium-length numbers with parentheses: 12345(20) - non-overlapping with long numbers
            r'(?<!\d)(\d{4,5}\(\d{1,3}\))(?!\d)',

            # 6-digit numbers with parentheses (separate pattern to avoid conflicts)
            r'(?<!\d)(\d{6}\(\d{1,3}\))(?!\d)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Enhanced validation for standalone mark numbers
                if self._is_high_confidence_mark_number(match, text):
                    mark_numbers.append(match)

        return mark_numbers

    def _is_high_confidence_mark_number(self, candidate: str, context: str) -> bool:
        """
        Enhanced validation for mark numbers with high confidence scoring.

        This method uses multiple criteria to determine if a candidate is likely a mark number.
        """
        if not candidate:
            return False

        confidence_score = 0

        # High confidence indicators

        # 1. Long numbers with parentheses (very strong indicator)
        if re.match(r'^\d{6,8}\(\d{1,3}\)$', candidate):
            confidence_score += 5

        # 2. Medium numbers with parentheses (strong indicator)
        elif re.match(r'^\d{4,6}\(\d{1,3}\)$', candidate):
            confidence_score += 4

        # 3. Decimal numbers (strong indicator for some invoice types)
        elif re.match(r'^\d{2,3}\.\d{1,2}$', candidate):
            confidence_score += 4

        # 4. Position analysis - appears on its own line or with minimal context
        if self._appears_standalone(candidate, context):
            confidence_score += 3

        # 5. Not surrounded by specification keywords
        if not self._surrounded_by_spec_keywords(candidate, context):
            confidence_score += 2

        # 6. Appears after line item data (positional indicator)
        if self._appears_after_line_item_data(candidate, context):
            confidence_score += 2

        # Negative indicators (reduce confidence)

        # 7. Too short or too long
        if len(candidate) < 4 or len(candidate) > 15:
            confidence_score -= 2

        # 8. Appears in the middle of a long specification line
        if self._appears_in_specification_line(candidate, context):
            confidence_score -= 3

        # Return True if confidence is high enough
        return confidence_score >= 6

    def _appears_standalone(self, candidate: str, context: str) -> bool:
        """Check if the candidate appears on its own line or with minimal context."""
        lines = context.split('\n')
        for line in lines:
            if candidate in line:
                line_clean = line.strip()
                # If the line is short and mostly contains the candidate
                if len(line_clean) <= len(candidate) + 10:
                    return True
                # If the candidate is at the end of a line with minimal other content
                if line_clean.endswith(candidate) and len(line_clean.replace(candidate, '').strip()) <= 20:
                    return True
        return False

    def _surrounded_by_spec_keywords(self, candidate: str, context: str) -> bool:
        """Check if the candidate is surrounded by specification keywords."""
        candidate_pos = context.find(candidate)
        if candidate_pos == -1:
            return False

        # Get surrounding context (30 characters before and after)
        start = max(0, candidate_pos - 30)
        end = min(len(context), candidate_pos + len(candidate) + 30)
        surrounding = context[start:end].upper()

        spec_keywords = [
            'GL', 'SU', 'RH', 'LH', 'EAP', 'FRAME', 'PIECE', 'DOOR',
            'MATERIAL', 'WELD', 'AMOUNT', 'CONFIGURED', 'SINGLE',
            'STEEL', 'ALUMINUM', 'GLASS', 'WINDOW', 'PANEL'
        ]

        keyword_count = sum(1 for keyword in spec_keywords if keyword in surrounding)
        return keyword_count >= 2  # If 2 or more spec keywords nearby, likely in specs

    def _appears_after_line_item_data(self, candidate: str, context: str) -> bool:
        """Check if the candidate appears after line item pricing data."""
        candidate_pos = context.find(candidate)
        if candidate_pos == -1:
            return False

        # Look at text before the candidate
        before_text = context[:candidate_pos]

        # Check if there are pricing patterns before this candidate
        pricing_patterns = [
            r'\d+\.\d{2,3}',  # Decimal prices like 914.000, 71.500
            r'\d{2,5}\.\d{2}',  # Amounts like 260.49, 7554.21
        ]

        for pattern in pricing_patterns:
            if re.search(pattern, before_text[-100:]):  # Check last 100 chars
                return True

        return False

    def _appears_in_specification_line(self, candidate: str, context: str) -> bool:
        """Check if the candidate appears in the middle of a specification line."""
        lines = context.split('\n')
        for line in lines:
            if candidate in line:
                line_clean = line.strip()
                # If the line is very long and has many specification-like elements
                if len(line_clean) > 80 and line_clean.count(' ') > 10:
                    # And the candidate is not at the end
                    if not line_clean.endswith(candidate):
                        return True
        return False

    def _extract_marks_by_position(self, all_lines: List[str], line_number: str) -> List[str]:
        """
        Extract mark numbers using positional/structural analysis.

        This method analyzes the document structure to find mark numbers that appear
        in consistent positions relative to line items, even without explicit labels.
        """
        mark_numbers = []

        # Find the line item and analyze surrounding context
        line_item_index = self._find_line_item_index(all_lines, line_number)
        if line_item_index == -1:
            return mark_numbers

        # Analyze lines following the line item (typically where mark numbers appear)
        context_lines = self._get_line_item_context(all_lines, line_item_index)

        for line in context_lines:
            # Extract potential mark numbers from each context line
            potential_marks = self._extract_potential_marks_from_line(line)
            mark_numbers.extend(potential_marks)

        return mark_numbers

    def _find_line_item_index(self, all_lines: List[str], line_number: str) -> int:
        """Find the index of the line item in the text."""
        for i, line in enumerate(all_lines):
            line_clean = line.strip()
            # More flexible line item detection
            if (line_clean.startswith(f"{line_number} ") or
                line_clean.startswith(f"{line_number}\t") or
                re.match(rf'^{line_number}\s+\d{{3}}\s+[A-Z0-9]+', line_clean)):
                return i
        return -1

    def _get_line_item_context(self, all_lines: List[str], line_item_index: int) -> List[str]:
        """Get the context lines that might contain mark numbers for a line item."""
        context_lines = []

        # Look at the next few lines after the line item
        for i in range(line_item_index + 1, min(line_item_index + 6, len(all_lines))):
            line = all_lines[i].strip()

            # Skip empty lines and separator lines
            if not line or line.startswith('---') or line.startswith('==='):
                continue

            # Stop if we hit another line item or major section
            if (re.match(r'^\d{3}\s+\d{3}\s+[A-Z0-9]+', line) or
                line.startswith('FREIGHT') or
                line.startswith('SHIPMENT') or
                line.startswith('TOTAL')):
                break

            context_lines.append(line)

        return context_lines

    def _extract_potential_marks_from_line(self, line: str) -> List[str]:
        """Extract potential mark numbers from a single line using enhanced analysis."""
        potential_marks = []

        # Use the enhanced pattern-based extraction on this line
        pattern_marks = self._extract_marks_by_patterns(line)
        potential_marks.extend(pattern_marks)

        # Additional sophisticated analysis for edge cases
        # Look for standalone codes that might be mark numbers
        words = line.split()
        for word in words:
            clean_word = word.strip('.,;:()[]{}')
            if clean_word and self._is_likely_mark_number_pattern(clean_word, line):
                # Additional validation to avoid false positives
                if self._is_high_confidence_mark_number(clean_word, line):
                    potential_marks.append(clean_word)

        return potential_marks

    def _is_likely_mark_number_pattern(self, candidate: str, context: str) -> bool:
        """
        Determine if a candidate string is likely to be a mark number.

        This is now more conservative to avoid false positives.
        """
        if not candidate or len(candidate) < 2 or len(candidate) > 25:
            return False

        # Use the same validation as the explicit mark number format
        return self._is_valid_mark_number_format(candidate)

    def _is_obviously_not_mark_number(self, candidate: str) -> bool:
        """Check if a candidate is obviously not a mark number."""
        # Decimal numbers (prices, measurements)
        if re.match(r'^\d+\.\d+$', candidate):
            return True

        # Small integers (quantities, line numbers we already have)
        if re.match(r'^\d{1,3}$', candidate) and int(candidate) < 1000:
            return True

        # Common words that aren't mark numbers
        common_words = {
            'FRAME', 'PIECE', 'DOOR', 'WINDOW', 'GLASS', 'STEEL', 'ALUMINUM',
            'LEFT', 'RIGHT', 'TOP', 'BOTTOM', 'SIDE', 'CENTER', 'MIDDLE',
            'INCH', 'FOOT', 'MM', 'CM', 'LB', 'KG', 'PSI', 'MPH',
            'RED', 'BLUE', 'GREEN', 'BLACK', 'WHITE', 'CLEAR',
            'FREIGHT', 'CHARGE', 'TOTAL', 'TAX', 'DISCOUNT', 'AMOUNT'
        }
        if candidate.upper() in common_words:
            return True

        # Very long strings (likely descriptions)
        if len(candidate) > 20:
            return True

        return False

    def _validate_and_clean_marks(self, mark_numbers: List[str], context: str) -> List[str]:
        """Validate and clean the extracted mark numbers."""
        if not mark_numbers:
            return []

        # Remove duplicates while preserving order and validate format
        seen = set()
        unique_marks = []
        for mark in mark_numbers:
            clean_mark = mark.strip()
            # Preserve original case for mark numbers (some might be lowercase)
            if clean_mark and clean_mark not in seen:
                # Use the precise validation
                if self._is_valid_mark_number_format(clean_mark):
                    seen.add(clean_mark)
                    unique_marks.append(clean_mark)

        return unique_marks



    def _extract_line_items_fallback(self, text: str, line_items: List[Dict[str, Any]]) -> None:
        """Fallback method to extract line items from unstructured text."""
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for lines that might be line items
            if self._looks_like_line_item(line):
                line_item = self._parse_line_item_from_line(line, i + 1)
                if line_item and self._validate_line_item(line_item):
                    line_items.append(line_item)

    def _looks_like_line_item(self, line: str) -> bool:
        """Check if a line looks like it contains line item data."""
        # Look for patterns that suggest line items
        patterns = [
            r'^\d{1,3}\s+\d{2,3}\s+[A-Z0-9]+',  # Line number, plant, item code
            r'^\d{1,3}\s+[A-Z0-9]+\s+\d+',      # Line number, item code, quantity
            r'\d+\s+\d+\s+[\d.,]+\s+[\d.,]+$',   # Quantities and prices at end
        ]

        return any(re.search(pattern, line) for pattern in patterns)

    def _parse_line_item_from_line(self, line: str, line_number: int) -> Dict[str, Any]:
        """Parse a single line into a line item."""
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
            "discount_amount": [
                r"discount\s*of\s*\$\s*([\d,]+\.?\d*)",
                r"discount\s*amount\s*:?\s*\$?([\d,]+\.?\d*)",
                r"total\s*discount\s*:?\s*\$?([\d,]+\.?\d*)",
                r"you\s*may\s*deduct\s*.*?\$?([\d,]+\.?\d*)"
            ],
            "total_sale": [
                r"total\s*sale\s*:?\s*\$?\s*([\d,]+\.?\d*)",
                r"subtotal\s*:?\s*\$?([\d,]+\.?\d*)",
                r"sub\s*total\s*:?\s*\$?([\d,]+\.?\d*)",
                r"net\s*amount\s*:?\s*\$?([\d,]+\.?\d*)"
            ],
            "tax": [
                r"tax\s*:?\s*\$?\s*([\d,]+\.?\d*)",
                r"sales\s*tax\s*:?\s*\$?([\d,]+\.?\d*)",
                r"vat\s*:?\s*\$?([\d,]+\.?\d*)"
            ],
            "invoice_total": [
                r"invoice\s*total\s*\(USD\)\s*:?\s*\$?\s*([\d,]+\.?\d*)",
                r"invoice\s*total\s*:?\s*\$?([\d,]+\.?\d*)",
                r"total\s*due\s*:?\s*\$?([\d,]+\.?\d*)",
                r"amount\s*due\s*:?\s*\$?([\d,]+\.?\d*)",
                r"grand\s*total\s*:?\s*\$?([\d,]+\.?\d*)",
                r"final\s*total\s*:?\s*\$?([\d,]+\.?\d*)"
            ],
            "freight": [
                r"freight\s*:?\s*\$?([\d,]+\.?\d*)",
                r"shipping\s*:?\s*\$?([\d,]+\.?\d*)",
                r"delivery\s*:?\s*\$?([\d,]+\.?\d*)"
            ]
        }

        for field, pattern_list in patterns.items():
            if isinstance(pattern_list, str):
                pattern_list = [pattern_list]

            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    # Clean and validate the amount
                    if value and re.match(r'^[\d,]+\.?\d*$', value):
                        totals_dict[field] = value
                    break

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
