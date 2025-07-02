#!/usr/bin/env python3
"""
Test script for the Enhanced Schlage PDF Processor
Validates accuracy improvements and comprehensive functionality
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
import time

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from schlage_processor import SchlageInvoiceProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchlageProcessorTester:
    """Comprehensive tester for the enhanced Schlage processor."""
    
    def __init__(self):
        """Initialize the tester."""
        self.processor = SchlageInvoiceProcessor(output_dir="output/schlage")
        self.test_results = []
        
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive tests on the enhanced processor."""
        logger.info("🚀 Starting comprehensive Schlage processor tests...")
        
        test_summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "accuracy_scores": [],
            "confidence_scores": [],
            "processing_times": [],
            "test_details": []
        }
        
        # Test 1: Process the existing Schlage PDF
        test_summary = self._test_existing_schlage_pdf(test_summary)
        
        # Test 2: Validate accuracy improvements
        test_summary = self._test_accuracy_improvements(test_summary)
        
        # Test 3: Test error handling and edge cases
        test_summary = self._test_error_handling(test_summary)
        
        # Test 4: Validate multi-format support
        test_summary = self._test_multi_format_support(test_summary)
        
        # Calculate final statistics
        test_summary["average_accuracy"] = sum(test_summary["accuracy_scores"]) / len(test_summary["accuracy_scores"]) if test_summary["accuracy_scores"] else 0
        test_summary["average_confidence"] = sum(test_summary["confidence_scores"]) / len(test_summary["confidence_scores"]) if test_summary["confidence_scores"] else 0
        test_summary["average_processing_time"] = sum(test_summary["processing_times"]) / len(test_summary["processing_times"]) if test_summary["processing_times"] else 0
        
        self._print_test_summary(test_summary)
        
        return test_summary
    
    def _test_existing_schlage_pdf(self, test_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Test processing of the existing Schlage PDF."""
        logger.info("📄 Test 1: Processing existing Schlage PDF...")
        
        pdf_path = Path("invoices/362259-027 Schlage 002.pdf")
        if not pdf_path.exists():
            logger.error(f"Test PDF not found: {pdf_path}")
            test_summary["failed_tests"] += 1
            test_summary["total_tests"] += 1
            return test_summary
        
        try:
            start_time = time.time()
            result = self.processor.process_schlage_pdf(pdf_path)
            processing_time = time.time() - start_time
            
            # Validate results
            accuracy = result.get("metadata", {}).get("accuracy_score", 0)
            confidence = result.get("validation", {}).get("overall_confidence", 0) * 100
            
            test_summary["accuracy_scores"].append(accuracy)
            test_summary["confidence_scores"].append(confidence)
            test_summary["processing_times"].append(processing_time)
            
            # Check if accuracy meets target
            if accuracy >= 90:
                test_summary["passed_tests"] += 1
                logger.info(f"✅ Test 1 PASSED: Accuracy {accuracy:.1f}% >= 90%")
            else:
                test_summary["failed_tests"] += 1
                logger.error(f"❌ Test 1 FAILED: Accuracy {accuracy:.1f}% < 90%")
            
            test_summary["test_details"].append({
                "test_name": "Existing Schlage PDF Processing",
                "status": "PASSED" if accuracy >= 90 else "FAILED",
                "accuracy": accuracy,
                "confidence": confidence,
                "processing_time": processing_time,
                "details": f"Processed {pdf_path.name} with {accuracy:.1f}% accuracy"
            })
            
        except Exception as e:
            logger.error(f"❌ Test 1 FAILED with exception: {e}")
            test_summary["failed_tests"] += 1
            test_summary["test_details"].append({
                "test_name": "Existing Schlage PDF Processing",
                "status": "FAILED",
                "error": str(e)
            })
        
        test_summary["total_tests"] += 1
        return test_summary
    
    def _test_accuracy_improvements(self, test_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Test accuracy improvements compared to baseline."""
        logger.info("📊 Test 2: Validating accuracy improvements...")
        
        # Load the original extracted data for comparison
        original_file = Path("output/schlage/362259-027 Schlage 002_extracted.json")
        new_file = Path("output/schlage/362259-027 Schlage 002_schlage_extracted.json")
        
        if not original_file.exists() or not new_file.exists():
            logger.warning("⚠️  Test 2 SKIPPED: Comparison files not found")
            test_summary["total_tests"] += 1
            return test_summary
        
        try:
            with open(original_file, 'r') as f:
                original_data = json.load(f)
            
            with open(new_file, 'r') as f:
                new_data = json.load(f)
            
            original_accuracy = original_data.get("metadata", {}).get("accuracy_score", 0)
            new_accuracy = new_data.get("metadata", {}).get("accuracy_score", 0)
            
            improvement = new_accuracy - original_accuracy
            
            if improvement >= 0:
                test_summary["passed_tests"] += 1
                logger.info(f"✅ Test 2 PASSED: Accuracy improved by {improvement:.1f}% ({original_accuracy:.1f}% → {new_accuracy:.1f}%)")
            else:
                test_summary["failed_tests"] += 1
                logger.error(f"❌ Test 2 FAILED: Accuracy decreased by {abs(improvement):.1f}%")
            
            test_summary["test_details"].append({
                "test_name": "Accuracy Improvement Validation",
                "status": "PASSED" if improvement >= 0 else "FAILED",
                "original_accuracy": original_accuracy,
                "new_accuracy": new_accuracy,
                "improvement": improvement,
                "details": f"Accuracy change: {improvement:+.1f}%"
            })
            
        except Exception as e:
            logger.error(f"❌ Test 2 FAILED with exception: {e}")
            test_summary["failed_tests"] += 1
            test_summary["test_details"].append({
                "test_name": "Accuracy Improvement Validation",
                "status": "FAILED",
                "error": str(e)
            })
        
        test_summary["total_tests"] += 1
        return test_summary
    
    def _test_error_handling(self, test_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Test error handling and robustness."""
        logger.info("🛡️  Test 3: Testing error handling...")
        
        # Test with non-existent file
        try:
            self.processor.process_schlage_pdf("non_existent_file.pdf")
            test_summary["failed_tests"] += 1
            logger.error("❌ Test 3a FAILED: Should have raised FileNotFoundError")
        except FileNotFoundError:
            test_summary["passed_tests"] += 1
            logger.info("✅ Test 3a PASSED: Correctly handled non-existent file")
        except Exception as e:
            test_summary["failed_tests"] += 1
            logger.error(f"❌ Test 3a FAILED: Unexpected exception: {e}")
        
        test_summary["total_tests"] += 1
        
        test_summary["test_details"].append({
            "test_name": "Error Handling - Non-existent File",
            "status": "PASSED",
            "details": "Correctly raised FileNotFoundError for non-existent file"
        })
        
        return test_summary
    
    def _test_multi_format_support(self, test_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Test multi-format support capabilities."""
        logger.info("🔄 Test 4: Testing multi-format support...")
        
        # This test validates that the processor can handle different extraction methods
        try:
            # Test the text extraction methods
            extractor = self.processor.text_extractor
            pdf_path = Path("invoices/362259-027 Schlage 002.pdf")
            
            if pdf_path.exists():
                text, confidence, method = extractor.extract_text_comprehensive(pdf_path)
                
                if text and confidence > 0:
                    test_summary["passed_tests"] += 1
                    logger.info(f"✅ Test 4 PASSED: Multi-format extraction successful (method: {method}, confidence: {confidence:.2f})")
                else:
                    test_summary["failed_tests"] += 1
                    logger.error("❌ Test 4 FAILED: No text extracted")
                
                test_summary["test_details"].append({
                    "test_name": "Multi-format Support",
                    "status": "PASSED" if text and confidence > 0 else "FAILED",
                    "extraction_method": method,
                    "confidence": confidence,
                    "text_length": len(text) if text else 0
                })
            else:
                logger.warning("⚠️  Test 4 SKIPPED: Test PDF not found")
                
        except Exception as e:
            test_summary["failed_tests"] += 1
            logger.error(f"❌ Test 4 FAILED with exception: {e}")
            test_summary["test_details"].append({
                "test_name": "Multi-format Support",
                "status": "FAILED",
                "error": str(e)
            })
        
        test_summary["total_tests"] += 1
        return test_summary
    
    def _print_test_summary(self, test_summary: Dict[str, Any]) -> None:
        """Print comprehensive test summary."""
        print("\n" + "="*70)
        print("🎯 ENHANCED SCHLAGE PROCESSOR TEST RESULTS")
        print("="*70)
        print(f"📊 Total Tests: {test_summary['total_tests']}")
        print(f"✅ Passed: {test_summary['passed_tests']}")
        print(f"❌ Failed: {test_summary['failed_tests']}")
        print(f"📈 Success Rate: {(test_summary['passed_tests']/test_summary['total_tests']*100):.1f}%")
        
        if test_summary["accuracy_scores"]:
            print(f"🎯 Average Accuracy: {test_summary['average_accuracy']:.1f}%")
            print(f"🔍 Average Confidence: {test_summary['average_confidence']:.1f}%")
            print(f"⏱️  Average Processing Time: {test_summary['average_processing_time']:.2f}s")
        
        print("\n📋 Detailed Test Results:")
        for i, test in enumerate(test_summary["test_details"], 1):
            status_icon = "✅" if test["status"] == "PASSED" else "❌"
            print(f"{i}. {status_icon} {test['test_name']}: {test['status']}")
            if "accuracy" in test:
                print(f"   📊 Accuracy: {test['accuracy']:.1f}%")
            if "details" in test:
                print(f"   📝 {test['details']}")
            if "error" in test:
                print(f"   ⚠️  Error: {test['error']}")
        
        print("\n" + "="*70)
        
        if test_summary["average_accuracy"] >= 90:
            print("🎉 SUCCESS: Enhanced processor meets 90%+ accuracy target!")
        else:
            print("⚠️  WARNING: Enhanced processor below 90% accuracy target")
        
        print("="*70)


def main():
    """Main function to run the comprehensive tests."""
    tester = SchlageProcessorTester()
    results = tester.run_comprehensive_tests()
    
    # Return appropriate exit code
    if results["failed_tests"] == 0 and results["average_accuracy"] >= 90:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
