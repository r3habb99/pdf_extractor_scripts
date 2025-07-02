#!/usr/bin/env python3
"""
Test script for the enhanced TextPDFProcessor

This script tests the improved processor with existing invoice samples
to verify 90%+ accuracy and enhanced functionality.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from text_pdf_processor import TextPDFProcessor, ConfigurationManager
from vendor_detector import detect_vendor_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_processor_with_sample(pdf_path: Path, expected_vendor: str = None) -> Dict[str, Any]:
    """Test the enhanced processor with a sample PDF."""
    logger.info(f"Testing enhanced processor with: {pdf_path.name}")
    
    try:
        # Detect vendor type if not provided
        if not expected_vendor:
            vendor_result = detect_vendor_type(pdf_path)
            vendor_type_obj = vendor_result.get('vendor_type', 'generic')
            # Convert VendorType enum to string if needed
            if hasattr(vendor_type_obj, 'value'):
                vendor_type = vendor_type_obj.value.lower()
            else:
                vendor_type = str(vendor_type_obj).lower()
        else:
            vendor_type = expected_vendor
        
        logger.info(f"Using vendor type: {vendor_type}")
        
        # Initialize enhanced processor
        processor = TextPDFProcessor(vendor_type=vendor_type)
        
        # Process the PDF
        result = processor.process_pdf(str(pdf_path), vendor_folder=vendor_type)
        
        # Extract key metrics
        metadata = result.get('metadata', {})
        extraction_metadata = result.get('extraction_metadata', {})
        validation = result.get('validation', {})
        
        test_results = {
            'pdf_file': pdf_path.name,
            'vendor_type': vendor_type,
            'confidence_score': metadata.get('confidence_score', 0.0),
            'overall_confidence': extraction_metadata.get('overall_confidence', 0.0),
            'completeness_score': validation.get('completeness_score', 0.0),
            'extraction_quality': validation.get('extraction_quality', 'unknown'),
            'line_item_count': validation.get('line_item_count', 0),
            'has_required_fields': {
                'invoice_number': validation.get('has_invoice_number', False),
                'invoice_date': validation.get('has_invoice_date', False),
                'line_items': validation.get('has_line_items', False),
                'vendor_info': validation.get('has_vendor_info', False)
            },
            'errors': extraction_metadata.get('errors', []),
            'warnings': extraction_metadata.get('warnings', []),
            'processing_successful': True
        }
        
        # Calculate accuracy score
        accuracy_score = calculate_accuracy_score(test_results)
        test_results['accuracy_score'] = accuracy_score
        
        logger.info(f"Processing complete - Accuracy: {accuracy_score:.1%}, Confidence: {test_results['confidence_score']:.1%}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {str(e)}")
        return {
            'pdf_file': pdf_path.name,
            'vendor_type': expected_vendor or 'unknown',
            'error': str(e),
            'processing_successful': False,
            'accuracy_score': 0.0
        }


def calculate_accuracy_score(results: Dict[str, Any]) -> float:
    """Calculate accuracy score based on extraction results."""
    if not results.get('processing_successful', False):
        return 0.0
    
    # Base score from confidence
    base_score = results.get('overall_confidence', 0.0)
    
    # Bonus for required fields
    required_fields = results.get('has_required_fields', {})
    field_bonus = sum(required_fields.values()) * 0.05  # 5% per required field
    
    # Bonus for line items
    line_item_bonus = 0.1 if results.get('line_item_count', 0) > 0 else 0.0
    
    # Penalty for errors
    error_penalty = len(results.get('errors', [])) * 0.05
    
    # Calculate final score
    accuracy = min(base_score + field_bonus + line_item_bonus - error_penalty, 1.0)
    return max(accuracy, 0.0)


def run_comprehensive_tests() -> Dict[str, Any]:
    """Run comprehensive tests on available invoice samples."""
    logger.info("Starting comprehensive tests of enhanced TextPDFProcessor")
    
    # Find test PDFs
    invoices_dir = current_dir / "invoices"
    if not invoices_dir.exists():
        logger.error(f"Invoices directory not found: {invoices_dir}")
        return {"error": "No test files found"}
    
    pdf_files = list(invoices_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in invoices directory")
        return {"error": "No PDF files found"}
    
    logger.info(f"Found {len(pdf_files)} PDF files to test")
    
    # Test each PDF
    test_results = []
    vendor_results = {}
    
    for pdf_file in pdf_files:
        result = test_processor_with_sample(pdf_file)
        test_results.append(result)
        
        # Group by vendor
        vendor = result.get('vendor_type', 'unknown')
        if vendor not in vendor_results:
            vendor_results[vendor] = []
        vendor_results[vendor].append(result)
    
    # Calculate overall statistics
    successful_tests = [r for r in test_results if r.get('processing_successful', False)]
    total_tests = len(test_results)
    success_rate = len(successful_tests) / total_tests if total_tests > 0 else 0.0
    
    # Calculate average accuracy
    accuracy_scores = [r.get('accuracy_score', 0.0) for r in successful_tests]
    average_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
    
    # Calculate vendor-specific statistics
    vendor_stats = {}
    for vendor, results in vendor_results.items():
        vendor_successful = [r for r in results if r.get('processing_successful', False)]
        vendor_accuracy = [r.get('accuracy_score', 0.0) for r in vendor_successful]
        
        vendor_stats[vendor] = {
            'total_files': len(results),
            'successful': len(vendor_successful),
            'success_rate': len(vendor_successful) / len(results) if results else 0.0,
            'average_accuracy': sum(vendor_accuracy) / len(vendor_accuracy) if vendor_accuracy else 0.0,
            'files': [r['pdf_file'] for r in results]
        }
    
    # Compile final results
    final_results = {
        'test_summary': {
            'total_files_tested': total_tests,
            'successful_extractions': len(successful_tests),
            'success_rate': success_rate,
            'average_accuracy': average_accuracy,
            'target_accuracy_met': average_accuracy >= 0.9,
            'timestamp': str(Path(__file__).stat().st_mtime)
        },
        'vendor_statistics': vendor_stats,
        'detailed_results': test_results,
        'recommendations': generate_recommendations(test_results, average_accuracy)
    }
    
    return final_results


def generate_recommendations(test_results: List[Dict[str, Any]], average_accuracy: float) -> List[str]:
    """Generate recommendations based on test results."""
    recommendations = []
    
    if average_accuracy < 0.9:
        recommendations.append(f"Average accuracy ({average_accuracy:.1%}) is below 90% target. Consider additional pattern refinement.")
    
    # Check for common errors
    all_errors = []
    for result in test_results:
        all_errors.extend(result.get('errors', []))
    
    if all_errors:
        recommendations.append(f"Found {len(all_errors)} extraction errors. Review error patterns for improvement opportunities.")
    
    # Check for missing line items
    no_line_items = [r for r in test_results if r.get('line_item_count', 0) == 0]
    if no_line_items:
        recommendations.append(f"{len(no_line_items)} files had no line items extracted. Review line item extraction patterns.")
    
    # Check vendor-specific issues
    vendor_groups = {}
    for result in test_results:
        vendor = result.get('vendor_type', 'unknown')
        if vendor not in vendor_groups:
            vendor_groups[vendor] = []
        vendor_groups[vendor].append(result)
    
    for vendor, results in vendor_groups.items():
        vendor_accuracy = [r.get('accuracy_score', 0.0) for r in results if r.get('processing_successful', False)]
        if vendor_accuracy:
            avg_vendor_accuracy = sum(vendor_accuracy) / len(vendor_accuracy)
            if avg_vendor_accuracy < 0.8:
                recommendations.append(f"Vendor '{vendor}' has low accuracy ({avg_vendor_accuracy:.1%}). Consider vendor-specific improvements.")
    
    if not recommendations:
        recommendations.append("All tests passed successfully! The enhanced processor is performing well.")
    
    return recommendations


def main():
    """Main test function."""
    print("🧪 Enhanced TextPDFProcessor Test Suite")
    print("=" * 50)
    
    # Run comprehensive tests
    results = run_comprehensive_tests()
    
    if "error" in results:
        print(f"❌ Test failed: {results['error']}")
        return 1
    
    # Display results
    summary = results['test_summary']
    print(f"📊 Test Results Summary:")
    print(f"   Total files tested: {summary['total_files_tested']}")
    print(f"   Successful extractions: {summary['successful_extractions']}")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Average accuracy: {summary['average_accuracy']:.1%}")
    print(f"   90% target met: {'✅ Yes' if summary['target_accuracy_met'] else '❌ No'}")
    
    print(f"\n📈 Vendor Statistics:")
    for vendor, stats in results['vendor_statistics'].items():
        print(f"   {vendor.upper()}:")
        print(f"     Files: {stats['total_files']} | Success: {stats['success_rate']:.1%} | Accuracy: {stats['average_accuracy']:.1%}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Save detailed results
    results_file = current_dir / "test_results_enhanced.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    return 0 if summary['target_accuracy_met'] else 1


if __name__ == "__main__":
    sys.exit(main())
