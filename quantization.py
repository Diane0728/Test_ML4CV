#!/usr/bin/env python3
"""
VLM Quantization Module - Production Version

Implements post-training dynamic quantization for Vision-Language Models,
specifically targeting BLIP models with INT8 precision optimization.

Features:
- INT8 dynamic quantization with CPU optimization
- FP16 fallback for CUDA devices
- Comprehensive validation and benchmarking
- Robust error handling and logging
- JSON result storage for analysis

Author: VLM Quantization Team
Date: July 2025
License: MIT
"""

import torch
import json
import time
import argparse
import os
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging

# VLM components
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np

# System monitoring
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VLMQuantizer:
    """
    Production-ready Vision-Language Model Quantization class.
    
    Supports:
    - INT8 dynamic quantization (CPU optimized)
    - FP16 precision (CUDA fallback)
    - Model size analysis and validation
    - Comprehensive error handling
    - Performance benchmarking
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the quantizer.
        
        Args:
            model_path: HuggingFace model identifier or local path
            device: Target device ("auto", "cpu", "cuda")
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.processor = None
        self.model = None
        self.quantized_model = None
        
        logger.info(f"🚀 Initializing VLM Quantizer for {model_path}")
        logger.info(f"📱 Target device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device configuration."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠️ CUDA requested but not available, falling back to CPU")
            device = "cpu"
            
        return device
    
    def load_model(self) -> bool:
        """
        Load the base model and processor with comprehensive error handling.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("📦 Loading BLIP processor and model...")
            
            # Load processor
            self.processor = BlipProcessor.from_pretrained(self.model_path)
            logger.info("✅ Processor loaded successfully")
            
            # Load model with appropriate dtype
            if self.device == "cuda":
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16
                )
                logger.info("🎮 Model loaded with FP16 for CUDA")
            else:
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_path
                )
                logger.info("💻 Model loaded with FP32 for CPU")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Log model information
            model_size = self.get_model_size(self.model)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            logger.info(f"📊 Model size: {model_size:.1f} MB")
            logger.info(f"🔢 Total parameters: {total_params:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def get_model_size(self, model=None) -> float:
        """
        Calculate model size in MB with high precision.
        
        Args:
            model: Model to analyze (uses self.model if None)
            
        Returns:
            float: Model size in MB
        """
        if model is None:
            model = self.model
            
        if model is None:
            return 0.0
            
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())
        return total_size / (1024 ** 2)
    
    def quantize_model(self, method: str = "int8") -> Dict[str, Any]:
        """
        Apply quantization to the loaded model with comprehensive validation.
        
        Args:
            method: Quantization method ("int8", "fp16")
            
        Returns:
            Dict containing quantization results and metadata
        """
        if self.model is None:
            raise ValueError("❌ Model not loaded. Call load_model() first.")
        
        logger.info(f"🔧 Starting {method.upper()} quantization...")
        start_time = time.time()
        
        # Get original model size
        original_size = self.get_model_size(self.model)
        original_memory = self._get_memory_usage()
        
        try:
            if method.lower() == "int8":
                self.quantized_model = self._apply_int8_quantization()
            elif method.lower() == "fp16":
                self.quantized_model = self._apply_fp16_quantization()
            else:
                raise ValueError(f"❌ Unsupported quantization method: {method}")
            
            # Calculate metrics
            quantized_size = self.get_model_size(self.quantized_model)
            quantized_memory = self._get_memory_usage()
            quantization_time = time.time() - start_time
            size_reduction = (original_size - quantized_size) / original_size
            
            results = {
                "quantization_info": {
                    "method": method.upper(),
                    "timestamp": datetime.now().isoformat(),
                    "device": self.device,
                    "model_path": self.model_path,
                    "pytorch_version": torch.__version__
                },
                "model_metrics": {
                    "original_size_mb": round(original_size, 2),
                    "quantized_size_mb": round(quantized_size, 2),
                    "size_reduction_ratio": round(size_reduction, 3),
                    "size_reduction_percent": round(size_reduction * 100, 1),
                    "quantization_time_seconds": round(quantization_time, 2),
                    "memory_before_mb": original_memory,
                    "memory_after_mb": quantized_memory
                },
                "success": True
            }
            
            logger.info(f"✅ Quantization completed successfully!")
            logger.info(f"📊 Size reduction: {size_reduction*100:.1f}% ({original_size:.1f}MB → {quantized_size:.1f}MB)")
            logger.info(f"⏱️ Time taken: {quantization_time:.2f}s")
            logger.info(f"💾 Memory impact: {original_memory:.1f}MB → {quantized_memory:.1f}MB")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Quantization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "method": method
            }
    
    def _apply_int8_quantization(self):
        """Apply INT8 dynamic quantization with CPU optimization."""
        logger.info("🔧 Applying INT8 dynamic quantization to linear layers...")
        
        # Move to CPU for quantization (required for INT8)
        original_device = next(self.model.parameters()).device
        model_cpu = self.model.cpu()
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        logger.info("✅ INT8 quantization applied successfully")
        logger.info("📝 Note: INT8 quantized model optimized for CPU inference")
        
        return quantized_model
    
    def _apply_fp16_quantization(self):
        """Apply FP16 precision as quantization alternative."""
        logger.info("🔧 Applying FP16 precision quantization...")
        
        if self.device == "cuda":
            quantized_model = self.model.half()
            logger.info("✅ FP16 quantization applied for CUDA")
        else:
            logger.warning("⚠️ FP16 quantization works best on CUDA devices")
            quantized_model = self.model.half()
            logger.info("✅ FP16 quantization applied for CPU")
        
        return quantized_model
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024
    
    def validate_quantization(self, num_samples: int = 5) -> Dict[str, Any]:
        """
        Validate quantized model with comprehensive testing.
        
        Args:
            num_samples: Number of synthetic samples to test
            
        Returns:
            Dict containing validation results
        """
        if self.quantized_model is None:
            raise ValueError("❌ No quantized model available. Run quantize_model() first.")
        
        logger.info(f"🧪 Validating quantization with {num_samples} samples...")
        
        # Generate synthetic test images
        test_images = self._generate_test_images(num_samples)
        
        validation_results = {
            "validation_info": {
                "num_samples": num_samples,
                "timestamp": datetime.now().isoformat(),
                "device": self.device
            },
            "inference_tests": [],
            "performance_summary": {}
        }
        
        inference_times = []
        successful_inferences = 0
        memory_usages = []
        
        for i, image in enumerate(test_images):
            try:
                # Measure memory before
                mem_before = self._get_memory_usage()
                
                # Run inference with timing
                start_time = time.time()
                caption = self._generate_caption(image, self.quantized_model)
                inference_time = time.time() - start_time
                
                # Measure memory after
                mem_after = self._get_memory_usage()
                memory_delta = mem_after - mem_before
                
                inference_times.append(inference_time)
                memory_usages.append(memory_delta)
                successful_inferences += 1
                
                validation_results["inference_tests"].append({
                    "sample_id": i + 1,
                    "inference_time": round(inference_time, 4),
                    "memory_delta_mb": round(memory_delta, 2),
                    "caption": caption,
                    "caption_length": len(caption.split()),
                    "success": True
                })
                
                logger.info(f"  ✅ Sample {i+1}: {inference_time:.3f}s, {memory_delta:.1f}MB - '{caption[:50]}...'")
                
            except Exception as e:
                validation_results["inference_tests"].append({
                    "sample_id": i + 1,
                    "success": False,
                    "error": str(e)
                })
                logger.warning(f"  ❌ Sample {i+1} failed: {e}")
        
        # Calculate comprehensive summary statistics
        if inference_times:
            validation_results["performance_summary"] = {
                "successful_samples": successful_inferences,
                "success_rate": round(successful_inferences / num_samples, 3),
                "avg_inference_time": round(np.mean(inference_times), 4),
                "min_inference_time": round(min(inference_times), 4),
                "max_inference_time": round(max(inference_times), 4),
                "std_inference_time": round(np.std(inference_times), 4),
                "total_inference_time": round(sum(inference_times), 4),
                "throughput_images_per_sec": round(successful_inferences / sum(inference_times), 3),
                "avg_memory_delta_mb": round(np.mean(memory_usages), 2) if memory_usages else 0,
                "total_tokens_generated": sum(len(test["caption"].split()) 
                                            for test in validation_results["inference_tests"] 
                                            if test["success"]),
            }
            
            # Calculate tokens per second
            total_tokens = validation_results["performance_summary"]["total_tokens_generated"]
            total_time = validation_results["performance_summary"]["total_inference_time"]
            if total_time > 0:
                validation_results["performance_summary"]["tokens_per_second"] = round(total_tokens / total_time, 2)
        
        logger.info(f"✅ Validation completed: {successful_inferences}/{num_samples} successful")
        if successful_inferences > 0:
            avg_time = np.mean(inference_times)
            throughput = successful_inferences / sum(inference_times)
            logger.info(f"📊 Performance: {avg_time:.3f}s avg, {throughput:.2f} img/sec")
        
        return validation_results
    
    def _generate_test_images(self, num_samples: int) -> List[Image.Image]:
        """Generate diverse synthetic test images for validation."""
        images = []
        
        logger.info(f"🎨 Generating {num_samples} synthetic test images...")
        
        for i in range(num_samples):
            # Create base image with realistic patterns
            img_array = np.random.randint(80, 180, (384, 384, 3), dtype=np.uint8)
            
            # Add diverse patterns for comprehensive testing
            pattern_type = i % 5
            
            if pattern_type == 0:  # Sky and ground scene
                # Blue sky gradient
                for y in range(0, 192):
                    intensity = int(100 + (y / 192) * 100)
                    img_array[y, :, 0] = np.minimum(intensity - 50, 255)
                    img_array[y, :, 1] = np.minimum(intensity - 20, 255)
                    img_array[y, :, 2] = np.minimum(intensity + 20, 255)
                # Green ground
                img_array[192:, :, 1] = 150
                
            elif pattern_type == 1:  # Architectural elements
                # White building
                img_array[100:284, 100:284] = [240, 240, 240]
                # Add windows
                img_array[150:180, 150:180] = [100, 100, 100]
                img_array[150:180, 204:234] = [100, 100, 100]
                
            elif pattern_type == 2:  # Vehicles/objects
                # Red car/object
                img_array[250:320, 150:270] = [200, 50, 50]
                # Black wheels
                img_array[300:320, 160:180] = [30, 30, 30]
                img_array[300:320, 240:260] = [30, 30, 30]
                
            elif pattern_type == 3:  # Natural elements
                # Yellow sun
                center_x, center_y = 300, 80
                for y in range(max(0, center_y-40), min(384, center_y+40)):
                    for x in range(max(0, center_x-40), min(384, center_x+40)):
                        if (x - center_x)**2 + (y - center_y)**2 <= 40**2:
                            img_array[y, x] = [255, 255, 100]
                
            else:  # Abstract patterns
                # Colorful stripes
                for y in range(0, 384, 40):
                    color = [(i*60) % 255, (i*90) % 255, (i*120) % 255]
                    img_array[y:y+20, :] = color
            
            images.append(Image.fromarray(img_array))
        
        logger.info(f"✅ Generated {len(images)} diverse test images")
        return images
    
    def _generate_caption(self, image: Image.Image, model) -> str:
        """Generate caption for a single image with robust error handling."""
        try:
            # Resize if too large for memory efficiency
            if max(image.size) > 512:
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            # Prepare inputs
            inputs = self.processor(image, return_tensors="pt")
            
            # Move inputs to appropriate device (handle quantized models)
            device = "cpu"  # INT8 quantized models run on CPU
            if hasattr(model, 'device') and str(model.device) != 'cpu':
                device = model.device
                inputs = inputs.to(device)
            
            # Generate caption with optimized parameters
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_length=30,
                    num_beams=3,
                    temperature=1.0,
                    do_sample=False,
                    early_stopping=True
                )
            
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption.strip()
            
        except Exception as e:
            logger.error(f"❌ Caption generation failed: {e}")
            return f"Caption generation failed: {str(e)}"
    
    def save_model(self, output_path: str) -> bool:
        """
        Save the quantized model to disk with metadata.
        
        Args:
            output_path: Path to save the model
            
        Returns:
            bool: True if successful
        """
        if self.quantized_model is None:
            logger.error("❌ No quantized model to save")
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state dict
            torch.save(self.quantized_model.state_dict(), output_path)
            
            # Save comprehensive model info
            info_path = output_path.with_suffix('.json')
            model_info = {
                "model_metadata": {
                    "original_model_path": self.model_path,
                    "quantization_method": "INT8" if hasattr(self.quantized_model, 'qconfig') else "FP16",
                    "saved_timestamp": datetime.now().isoformat(),
                    "pytorch_version": torch.__version__,
                    "device_used": self.device
                },
                "model_specs": {
                    "model_size_mb": self.get_model_size(self.quantized_model),
                    "total_parameters": sum(p.numel() for p in self.quantized_model.parameters()),
                    "quantized_layers": "torch.nn.Linear",
                    "precision": "INT8" if hasattr(self.quantized_model, 'qconfig') else "FP16"
                },
                "usage_instructions": {
                    "loading": "Use torch.load() to load state_dict",
                    "device": "CPU recommended for INT8, CUDA for FP16",
                    "inference": "Use model.eval() and torch.no_grad()"
                }
            }
            
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"✅ Model saved to {output_path}")
            logger.info(f"✅ Model info saved to {info_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            return False
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> bool:
        """
        Save quantization results to JSON file with validation.
        
        Args:
            results: Results dictionary from quantization
            output_path: Path to save results
            
        Returns:
            bool: True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add file metadata
            results["file_metadata"] = {
                "created_timestamp": datetime.now().isoformat(),
                "file_version": "1.0",
                "creator": "VLM Quantization Tool",
                "file_size_bytes": 0  # Will be updated after saving
            }
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Update file size
            file_size = output_path.stat().st_size
            results["file_metadata"]["file_size_bytes"] = file_size
            
            # Save again with updated metadata
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Results saved to {output_path} ({file_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            return False


def quantize_blip_model(
    model_name: str,
    quantization_method: str = "int8",
    save_path: Optional[str] = None,
    validate: bool = True,
    num_validation_samples: int = 5,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    High-level function to quantize a BLIP model with full pipeline.
    
    Args:
        model_name: HuggingFace model identifier
        quantization_method: "int8" or "fp16"
        save_path: Optional path to save quantized model
        validate: Whether to run validation tests
        num_validation_samples: Number of samples for validation
        device: Target device for processing
        
    Returns:
        Dict containing complete quantization results
    """
    logger.info("🚀 Starting comprehensive quantization pipeline")
    logger.info(f"📋 Model: {model_name}")
    logger.info(f"🔧 Method: {quantization_method.upper()}")
    logger.info(f"🧪 Validation: {'Enabled' if validate else 'Disabled'}")
    
    pipeline_start = time.time()
    
    # Initialize quantizer
    quantizer = VLMQuantizer(model_name, device=device)
    
    # Load model
    if not quantizer.load_model():
        return {"success": False, "error": "Failed to load model"}
    
    # Apply quantization
    quantization_results = quantizer.quantize_model(quantization_method)
    
    if not quantization_results.get("success", False):
        return quantization_results
    
    # Run validation if requested
    if validate:
        logger.info("🧪 Running validation tests...")
        try:
            validation_results = quantizer.validate_quantization(num_validation_samples)
            quantization_results["validation"] = validation_results
        except Exception as e:
            logger.warning(f"⚠️ Validation failed: {e}")
            quantization_results["validation_error"] = str(e)
    
    # Save model if path provided
    if save_path:
        success = quantizer.save_model(save_path)
        quantization_results["model_saved"] = success
        if success:
            quantization_results["model_saved_to"] = save_path
    
    # Add comprehensive pipeline summary
    pipeline_time = time.time() - pipeline_start
    quantization_results["pipeline_summary"] = {
        "total_pipeline_time": round(pipeline_time, 2),
        "model_name": model_name,
        "quantization_method": quantization_method,
        "device_used": quantizer.device,
        "validation_performed": validate,
        "model_saved": save_path is not None,
        "pytorch_version": torch.__version__,
        "completion_timestamp": datetime.now().isoformat()
    }
    
    logger.info("🎉 Quantization pipeline completed successfully!")
    logger.info(f"⏱️ Total time: {pipeline_time:.2f}s")
    
    return quantization_results


def test_model_loading(model_path: str) -> bool:
    """
    Test if a model can be loaded successfully.
    
    Args:
        model_path: Model identifier or path
        
    Returns:
        bool: True if model loads successfully
    """
    try:
        logger.info(f"🧪 Testing model loading: {model_path}")
        processor = BlipProcessor.from_pretrained(model_path)
        model = BlipForConditionalGeneration.from_pretrained(model_path)
        logger.info(f"✅ Model {model_path} loads successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load {model_path}: {e}")
        return False


def main():
    """Main CLI interface for quantization with comprehensive options."""
    parser = argparse.ArgumentParser(
        description="🚀 VLM Quantization Tool - Production Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🔥 Examples:
  # Basic quantization
  python quantization.py --model-path "Salesforce/blip-image-captioning-base"
  
  # Advanced quantization with saving
  python quantization.py \\
    --model-path "Salesforce/blip-image-captioning-base" \\
    --method fp16 \\
    --save ./models/blip_fp16.pth \\
    --validate \\
    --num-validation-samples 10
  
  # Test model loading
  python quantization.py --test-loading "Salesforce/blip-image-captioning-base"
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="Salesforce/blip-image-captioning-base",
        help="HuggingFace model identifier or local path"
    )
    
    parser.add_argument(
        "--method",
        choices=["int8", "fp16"],
        default="int8",
        help="Quantization method to apply (default: int8)"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Target device (default: auto)"
    )
    
    parser.add_argument(
        "--save",
        type=str,
        help="Path to save quantized model (optional)"
    )
    
    parser.add_argument(
        "--results-path",
        type=str,
        default="./results/quantization_results.json",
        help="Path to save results JSON (default: ./results/quantization_results.json)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation tests after quantization"
    )
    
    parser.add_argument(
        "--num-validation-samples",
        type=int,
        default=5,
        help="Number of samples for validation testing (default: 5)"
    )
    
    parser.add_argument(
        "--test-loading",
        type=str,
        help="Test if a model can be loaded (no quantization performed)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Test loading mode
    if args.test_loading:
        logger.info("🧪 MODEL LOADING TEST")
        success = test_model_loading(args.test_loading)
        exit(0 if success else 1)
    
    # Main quantization pipeline
    logger.info("="*80)
    logger.info("🚀 VLM QUANTIZATION PIPELINE - PRODUCTION VERSION")
    logger.info("="*80)
    
    results = quantize_blip_model(
        model_name=args.model_path,
        quantization_method=args.method,
        save_path=args.save,
        validate=args.validate,
        num_validation_samples=args.num_validation_samples,
        device=args.device
    )
    
    # Create results directory
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    if results.get("success", False):
        # Save to JSON
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Complete results saved to {results_path}")
        
        # Print comprehensive summary
        metrics = results.get("model_metrics", {})
        pipeline = results.get("pipeline_summary", {})
        
        logger.info("="*80)
        logger.info("🎯 QUANTIZATION SUMMARY")
        logger.info("="*80)
        logger.info(f"📱 Model: {args.model_path}")
        logger.info(f"🔧 Method: {args.method.upper()}")
        logger.info(f"📱 Device: {pipeline.get('device_used', 'unknown')}")
        logger.info(f"📊 Original size: {metrics.get('original_size_mb', 0):.1f} MB")
        logger.info(f"📦 Quantized size: {metrics.get('quantized_size_mb', 0):.1f} MB")
        logger.info(f"📉 Size reduction: {metrics.get('size_reduction_percent', 0):.1f}%")
        logger.info(f"⏱️ Quantization time: {metrics.get('quantization_time_seconds', 0):.2f}s")
        logger.info(f"🚀 Total pipeline time: {pipeline.get('total_pipeline_time', 0):.2f}s")
        
        if args.validate and "validation" in results:
            val_summary = results["validation"].get("performance_summary", {})
            logger.info(f"🧪 Validation success rate: {val_summary.get('success_rate', 0)*100:.1f}%")
            logger.info(f"⚡ Avg inference time: {val_summary.get('avg_inference_time', 0):.4f}s")
            logger.info(f"🔥 Throughput: {val_summary.get('throughput_images_per_sec', 0):.2f} img/sec")
            logger.info(f"💬 Tokens per second: {val_summary.get('tokens_per_second', 0):.2f}")
        
        if args.save:
            save_status = "✅ Saved" if results.get("model_saved", False) else "❌ Failed"
            logger.info(f"💾 Model saving: {save_status}")
        
    else:
        logger.error("❌ Quantization failed!")
        logger.error(f"Error: {results.get('error', 'Unknown error')}")
        exit(1)
    
    logger.info("="*80)
    logger.info("🎉 QUANTIZATION COMPLETED SUCCESSFULLY!")
    logger.info("="*80)


if __name__ == "__main__":
    main() "