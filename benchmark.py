#!/usr/bin/env python3
"""
VLM Benchmark Module - Production Version

Comprehensive benchmarking suite for Vision-Language Models with support for
quantized model evaluation, performance analysis, and comparative studies.

Features:
- Multi-metric performance evaluation
- Quality assessment with BLEU and custom metrics
- Memory and speed profiling
- Comparative analysis framework
- Production-ready error handling
- JSON result storage and visualization

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
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

# Core ML libraries
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# VLM components
from transformers import BlipProcessor, BlipForConditionalGeneration

# Evaluation metrics with fallbacks
try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# System monitoring
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VLMBenchmarker:
    """
    Production-ready benchmarking suite for Vision-Language Models.
    
    Features:
    - Comprehensive performance metrics (speed, memory, throughput)
    - Multi-tier quality evaluation (BLEU, NLTK, custom similarity)
    - Comparative analysis between models
    - Robust error handling and fallback mechanisms
    - Professional result visualization and reporting
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize benchmarker with comprehensive setup.
        
        Args:
            device: Target device ("auto", "cpu", "cuda")
        """
        self.device = self._setup_device(device)
        self.processor = None
        self.model = None
        
        # Performance tracking
        self.inference_times = []
        self.memory_usage = []
        self.predictions = []
        self.references = []
        
        # Benchmark metadata
        self.benchmark_start_time = None
        self.model_info = {}
        
        logger.info(f"🚀 VLM Benchmarker initialized on {self.device}")
        logger.info(f"📊 Evaluation methods available: HF-Evaluate={EVALUATE_AVAILABLE}, NLTK={NLTK_AVAILABLE}")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device configuration."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠️ CUDA requested but not available, falling back to CPU")
            device = "cpu"
            
        return device
    
    def load_model(self, model_path: str, quantized: bool = False) -> bool:
        """
        Load model for benchmarking with comprehensive validation.
        
        Args:
            model_path: Model identifier or path
            quantized: Whether this is a quantized model
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"📦 Loading {'quantized' if quantized else 'standard'} model: {model_path}")
            
            # Load processor
            if quantized and os.path.exists(model_path) and model_path.endswith('.pth'):
                # For quantized models, use base model processor
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                logger.info("✅ Using base model processor for quantized model")
            else:
                self.processor = BlipProcessor.from_pretrained(model_path)
                logger.info("✅ Processor loaded successfully")
            
            # Load model based on type
            if quantized and os.path.exists(model_path) and model_path.endswith('.pth'):
                # Load quantized model from saved state
                base_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                base_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model = base_model
                logger.info("✅ Quantized model loaded from saved state")
            else:
                # Load standard model
                if self.device == "cuda":
                    self.model = BlipForConditionalGeneration.from_pretrained(
                        model_path, torch_dtype=torch.float16
                    )
                else:
                    self.model = BlipForConditionalGeneration.from_pretrained(model_path)
                logger.info("✅ Standard model loaded")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Store model information
            self.model_info = {
                "model_path": model_path,
                "is_quantized": quantized,
                "device": self.device,
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "model_size_mb": self._get_model_size(),
                "model_type": type(self.model).__name__,
                "load_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"📊 Parameters: {self.model_info['total_parameters']:,}")
            logger.info(f"💾 Size: {self.model_info['model_size_mb']:.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def _get_model_size(self) -> float:
        """Calculate model size in MB."""
        if self.model is None:
            return 0.0
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)
    
    def generate_caption(self, image: Image.Image, max_length: int = 50) -> Tuple[str, float, float]:
        """
        Generate caption with comprehensive timing and memory tracking.
        
        Args:
            image: Input image
            max_length: Maximum caption length
            
        Returns:
            Tuple of (caption, inference_time, memory_delta)
        """
        try:
            # Resize large images for memory efficiency
            original_size = image.size
            if max(image.size) > 512:
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            # Measure memory before
            memory_before = self._get_memory_usage()
            
            # Prepare inputs
            inputs = self.processor(image, return_tensors="pt")
            
            # Handle device placement for different model types
            if hasattr(self.model, 'device') and str(self.model.device) != 'cpu':
                inputs = inputs.to(self.model.device)
            elif self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # Time inference with high precision
            start_time = time.perf_counter()
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=3,
                    temperature=1.0,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            inference_time = time.perf_counter() - start_time
            
            # Measure memory after
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before
            
            # Decode caption
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            return caption.strip(), inference_time, memory_delta
            
        except Exception as e:
            logger.error(f"❌ Caption generation failed: {e}")
            return "Caption generation failed", 0.0, 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024
    
    def benchmark_performance(
        self,
        images: List[Image.Image],
        references: Optional[List[List[str]]] = None,
        batch_size: int = 1,
        warmup_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmark with detailed analysis.
        
        Args:
            images: List of test images
            references: Optional reference captions for quality evaluation
            batch_size: Processing batch size (currently supports 1)
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dict containing comprehensive benchmark results
        """
        logger.info(f"🚀 Starting comprehensive benchmark with {len(images)} images")
        logger.info(f"🔥 Warmup iterations: {warmup_iterations}")
        
        self.benchmark_start_time = time.time()
        
        # Reset tracking lists
        self.inference_times = []
        self.memory_usage = []
        self.predictions = []
        self.references = references or []
        
        # Model warmup
        logger.info("🔥 Warming up model...")
        if images:
            for i in range(warmup_iterations):
                _, _, _ = self.generate_caption(images[0])
                logger.debug(f"  Warmup iteration {i+1}/{warmup_iterations} completed")
        
        # Clear cache after warmup
        self._clear_memory_cache()
        logger.info("✅ Warmup completed, memory cleared")
        
        # Main benchmark loop
        logger.info("📊 Running main benchmark...")
        successful_inferences = 0
        failed_inferences = 0
        total_tokens = 0
        
        start_time = time.time()
        
        for i, image in enumerate(tqdm(images, desc="🔄 Processing images", unit="img")):
            try:
                # Generate caption with metrics
                caption, inference_time, memory_delta = self.generate_caption(image)
                
                # Validate results
                if inference_time > 0 and caption != "Caption generation failed":
                    self.predictions.append(caption)
                    self.inference_times.append(inference_time)
                    self.memory_usage.append(memory_delta)
                    total_tokens += len(caption.split())
                    successful_inferences += 1
                else:
                    failed_inferences += 1
                    logger.warning(f"⚠️ Failed inference for image {i+1}")
                
                # Periodic cleanup and progress update
                if (i + 1) % 10 == 0:
                    self._clear_memory_cache()
                    avg_time = np.mean(self.inference_times[-10:]) if self.inference_times else 0
                    logger.info(f"📈 Processed {i+1}/{len(images)}, avg time (last 10): {avg_time:.3f}s")
                
            except Exception as e:
                logger.warning(f"❌ Error processing image {i+1}: {e}")
                failed_inferences += 1
                continue
        
        total_benchmark_time = time.time() - start_time
        
        # Calculate comprehensive metrics
        results = self._calculate_comprehensive_metrics(
            total_benchmark_time, len(images), successful_inferences, 
            failed_inferences, total_tokens
        )
        
        # Add quality metrics if references provided
        if self.references and self.predictions:
            logger.info("📊 Calculating quality metrics...")
            quality_metrics = self._calculate_quality_metrics()
            results.update(quality_metrics)
        
        # Add sample predictions for analysis
        results["sample_outputs"] = {
            "first_5_predictions": self.predictions[:5],
            "last_5_predictions": self.predictions[-5:] if len(self.predictions) > 5 else [],
            "random_samples": self._get_random_samples(3)
        }
        
        logger.info("✅ Benchmark completed successfully!")
        return results
    
    def _clear_memory_cache(self):
        """Clear memory cache for both CPU and GPU."""
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _calculate_comprehensive_metrics(
        self, total_time: float, total_images: int, successful: int, 
        failed: int, total_tokens: int
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        
        # Filter valid measurements
        valid_times = [t for t in self.inference_times if t > 0]
        valid_memory = [m for m in self.memory_usage if m is not None and not np.isnan(m)]
        
        if not valid_times:
            logger.error("❌ No valid inference times recorded")
            return {"error": "No valid inferences completed"}
        
        # Core performance metrics
        performance_metrics = {
            "total_samples": total_images,
            "successful_samples": successful,
            "failed_samples": failed,
            "success_rate": round(successful / total_images, 3),
            "total_benchmark_time": round(total_time, 3),
            "avg_inference_time": round(np.mean(valid_times), 4),
            "min_inference_time": round(min(valid_times), 4),
            "max_inference_time": round(max(valid_times), 4),
            "std_inference_time": round(np.std(valid_times), 4),
            "median_inference_time": round(np.median(valid_times), 4),
            "p95_inference_time": round(np.percentile(valid_times, 95), 4),
            "p99_inference_time": round(np.percentile(valid_times, 99), 4),
            "throughput_images_per_sec": round(successful / total_time, 3),
            "tokens_per_second": round(total_tokens / total_time, 2),
            "avg_tokens_per_prediction": round(total_tokens / successful, 2) if successful > 0 else 0,
            "total_tokens_generated": total_tokens
        }
        
        # Memory metrics
        if valid_memory:
            performance_metrics.update({
                "avg_memory_usage_mb": round(np.mean(valid_memory), 2),
                "max_memory_usage_mb": round(max(valid_memory), 2),
                "min_memory_usage_mb": round(min(valid_memory), 2),
                "std_memory_usage_mb": round(np.std(valid_memory), 2)
            })
        
        # Statistical analysis
        performance_metrics["statistical_analysis"] = {
            "coefficient_of_variation": round(np.std(valid_times) / np.mean(valid_times), 4),
            "efficiency_score": round(successful / total_time, 3),  # Images per second
            "stability_score": round(1 - (np.std(valid_times) / np.mean(valid_times)), 3)
        }
        
        return {
            "performance_metrics": performance_metrics,
            "system_info": {
                "device": self.device,
                "cuda_available": torch.cuda.is_available(),
                "pytorch_version": torch.__version__,
                "timestamp": datetime.now().isoformat(),
                "benchmark_duration": round(time.time() - self.benchmark_start_time, 2)
            },
            "model_info": self.model_info
        }
    
    def _get_random_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Get random sample predictions for analysis."""
        if not self.predictions:
            return []
        
        indices = np.random.choice(len(self.predictions), 
                                 min(num_samples, len(self.predictions)), 
                                 replace=False)
        
        return [
            {
                "index": int(idx),
                "prediction": self.predictions[idx],
                "inference_time": self.inference_times[idx] if idx < len(self.inference_times) else None,
                "memory_usage": self.memory_usage[idx] if idx < len(self.memory_usage) else None
            }
            for idx in indices
        ]
    
    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate quality metrics with comprehensive fallback system."""
        
        if not self.references or not self.predictions:
            return {"quality_metrics": {"error": "No references or predictions available"}}
        
        logger.info("📊 Calculating quality metrics with multi-tier evaluation...")
        
        quality_metrics = {
            "evaluation_info": {
                "num_predictions": len(self.predictions),
                "num_references": len(self.references),
                "evaluation_timestamp": datetime.now().isoformat()
            }
        }
        
        # Tier 1: HuggingFace Evaluate (preferred)
        if EVALUATE_AVAILABLE:
            try:
                bleu = evaluate.load("bleu")
                result = bleu.compute(predictions=self.predictions, references=self.references)
                quality_metrics["bleu_score"] = round(result['bleu'], 4)
                quality_metrics["evaluation_method"] = "HuggingFace Evaluate"
                quality_metrics["bleu_details"] = {
                    "bleu_1": round(result.get('precisions', [0])[0] if result.get('precisions') else 0, 4),
                    "bleu_2": round(result.get('precisions', [0, 0])[1] if len(result.get('precisions', [])) > 1 else 0, 4),
                    "bleu_3": round(result.get('precisions', [0, 0, 0])[2] if len(result.get('precisions', [])) > 2 else 0, 4),
                    "bleu_4": round(result.get('precisions', [0, 0, 0, 0])[3] if len(result.get('precisions', [])) > 3 else 0, 4),
                    "brevity_penalty": round(result.get('brevity_penalty', 0), 4)
                }
                logger.info("✅ Used HuggingFace Evaluate for comprehensive BLEU analysis")
            except Exception as e:
                logger.warning(f"⚠️ HuggingFace Evaluate failed: {e}")
        
        # Tier 2: NLTK BLEU (fallback)
        if "bleu_score" not in quality_metrics and NLTK_AVAILABLE:
            try:
                from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
                
                # Convert to NLTK format
                references_nltk = []
                for refs in self.references:
                    if isinstance(refs, list):
                        refs_tokens = [ref.split() for ref in refs]
                    else:
                        refs_tokens = [refs.split()]
                    references_nltk.append(refs_tokens)
                
                predictions_nltk = [pred.split() for pred in self.predictions]
                
                # Calculate corpus BLEU
                bleu_score = corpus_bleu(references_nltk, predictions_nltk)
                quality_metrics["bleu_score"] = round(bleu_score, 4)
                quality_metrics["evaluation_method"] = "NLTK"
                
                # Calculate individual BLEU scores for analysis
                individual_scores = []
                for ref, pred in zip(references_nltk[:10], predictions_nltk[:10]):  # Sample first 10
                    try:
                        score = sentence_bleu(ref, pred)
                        individual_scores.append(score)
                    except:
                        continue
                
                if individual_scores:
                    quality_metrics["bleu_statistics"] = {
                        "mean_individual_bleu": round(np.mean(individual_scores), 4),
                        "std_individual_bleu": round(np.std(individual_scores), 4),
                        "sample_size": len(individual_scores)
                    }
                
                logger.info("✅ Used NLTK for BLEU calculation")
                
            except Exception as e:
                logger.warning(f"⚠️ NLTK BLEU failed: {e}")
        
        # Tier 3: Custom similarity (final fallback)
        if "bleu_score" not in quality_metrics:
            try:
                similarity_score = self._calculate_custom_similarity()
                quality_metrics["similarity_score"] = round(similarity_score, 4)
                quality_metrics["evaluation_method"] = "Custom Word Overlap Similarity"
                logger.info("✅ Used custom similarity metric as fallback")
            except Exception as e:
                logger.error(f"❌ All quality metrics failed: {e}")
                quality_metrics["evaluation_method"] = "Failed"
                quality_metrics["error"] = str(e)
        
        # Additional quality analysis
        if self.predictions:
            quality_metrics["caption_analysis"] = {
                "avg_caption_length": round(np.mean([len(pred.split()) for pred in self.predictions]), 2),
                "min_caption_length": min(len(pred.split()) for pred in self.predictions),
                "max_caption_length": max(len(pred.split()) for pred in self.predictions),
                "unique_captions": len(set(self.predictions)),
                "repetition_rate": round(1 - (len(set(self.predictions)) / len(self.predictions)), 3)
            }
        
        return {"quality_metrics": quality_metrics}
    
    def _calculate_custom_similarity(self) -> float:
        """Calculate custom word overlap similarity metric."""
        
        total_similarity = 0
        count = 0
        
        for pred, refs in zip(self.predictions, self.references):
            if isinstance(refs, list):
                ref = refs[0] if refs else ""
            else:
                ref = refs
            
            pred_words = set(str(pred).lower().split())
            ref_words = set(str(ref).lower().split())
            
            if len(pred_words) == 0 and len(ref_words) == 0:
                similarity = 1.0
            elif len(pred_words) == 0 or len(ref_words) == 0:
                similarity = 0.0
            else:
                intersection = len(pred_words.intersection(ref_words))
                union = len(pred_words.union(ref_words))
                similarity = intersection / union if union > 0 else 0.0
            
            total_similarity += similarity
            count += 1
        
        return total_similarity / count if count > 0 else 0.0
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> bool:
        """
        Save benchmark results to JSON file with comprehensive metadata.
        
        Args:
            results: Results dictionary
            output_path: Path to save results
            
        Returns:
            bool: True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add comprehensive metadata
            results["benchmark_metadata"] = {
                "version": "2.0",
                "tool": "VLM Benchmarker - Production Version",
                "timestamp": datetime.now().isoformat(),
                "total_predictions": len(self.predictions),
                "device_used": self.device,
                "pytorch_version": torch.__version__,
                "evaluation_libraries": {
                    "huggingface_evaluate": EVALUATE_AVAILABLE,
                    "nltk": NLTK_AVAILABLE
                },
                "file_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save with pretty formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Log file info
            file_size = output_path.stat().st_size
            logger.info(f"✅ Results saved to {output_path}")
            logger.info(f"📄 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            return False


# Utility functions for data loading and processing

def load_test_images(images_dir: str, max_images: int = 50) -> List[Image.Image]:
    """
    Load test images from directory with comprehensive error handling.
    
    Args:
        images_dir: Directory containing images
        max_images: Maximum number of images to load
        
    Returns:
        List of PIL Images
    """
    logger.info(f"📁 Loading test images from {images_dir}")
    
    if not os.path.exists(images_dir):
        logger.warning(f"⚠️ Images directory not found: {images_dir}")
        return []
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    images = []
    failed_loads = 0
    
    for img_path in sorted(image_files)[:max_images]:
        try:
            image = Image.open(img_path).convert('RGB')
            images.append(image)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {img_path}: {e}")
            failed_loads += 1
    
    logger.info(f"✅ Loaded {len(images)} images successfully")
    if failed_loads > 0:
        logger.warning(f"⚠️ Failed to load {failed_loads} images")
    
    return images


def load_coco_annotations(annotations_file: str, images_dir: str, max_samples: int = 50) -> Tuple[List[Image.Image], List[List[str]]]:
    """
    Load COCO dataset with annotations and comprehensive validation.
    
    Args:
        annotations_file: Path to COCO annotations JSON
        images_dir: Directory containing COCO images
        max_samples: Maximum samples to load
        
    Returns:
        Tuple of (images, references)
    """
    logger.info(f"📋 Loading COCO data from {annotations_file}")
    
    try:
        with open(annotations_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        logger.info(f"📊 COCO data loaded: {len(coco_data.get('images', []))} images, {len(coco_data.get('annotations', []))} annotations")
        
        # Create image to captions mapping
        image_captions = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_captions:
                image_captions[img_id] = []
            image_captions[img_id].append(ann['caption'])
        
        # Get image filenames
        image_files = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Load images and references
        images = []
        references = []
        failed_loads = 0
        
        for img_id, filename in image_files.items():
            if len(images) >= max_samples:
                break
                
            img_path = os.path.join(images_dir, filename)
            if os.path.exists(img_path) and img_id in image_captions:
                try:
                    image = Image.open(img_path).convert('RGB')
                    images.append(image)
                    references.append(image_captions[img_id])
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {img_path}: {e}")
                    failed_loads += 1
        
        logger.info(f"✅ Loaded {len(images)} COCO samples with references")
        if failed_loads > 0:
            logger.warning(f"⚠️ Failed to load {failed_loads} images")
            
        return images, references
        
    except Exception as e:
        logger.error(f"❌ Failed to load COCO data: {e}")
        return [], []


def create_synthetic_dataset(num_samples: int = 10) -> Tuple[List[Image.Image], List[List[str]]]:
    """
    Create diverse synthetic test dataset with realistic patterns.
    
    Args:
        num_