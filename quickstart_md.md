# Quick Start Guide

Get up and running with VLM quantization in under 5 minutes!

## 🔧 Installation

### Option 1: Standard Installation
```bash
git clone https://github.com/your-username/vlm-quantization-benchmark.git
cd vlm-quantization-benchmark
pip install -r requirements.txt
```

### Option 2: Development Installation
```bash
git clone https://github.com/your-username/vlm-quantization-benchmark.git
cd vlm-quantization-benchmark
pip install -e .
pip install -r requirements.txt
```

### Option 3: Google Colab
```python
!git clone https://github.com/your-username/vlm-quantization-benchmark.git
%cd vlm-quantization-benchmark
!pip install -r requirements.txt
```

## 🚀 Basic Usage

### 1. Simple Quantization Example
```python
from quantization import quantize_blip_model

# Quick quantization with default settings
results = quantize_blip_model(
    model_name="Salesforce/blip-image-captioning-base",
    save_path="./models/blip_quantized.pth"
)

print(f"Model size reduced by {results['size_reduction']:.1%}")
print(f"Quantization completed in {results['quantization_time']:.2f}s")
```

### 2. Run Complete Benchmark
```bash
# Basic benchmark with 25 samples
python benchmark.py \
    --model-path "Salesforce/blip-image-captioning-base" \
    --num-samples 25 \
    --output-dir ./results

# Advanced benchmark with custom dataset
python benchmark.py \
    --model-path "Salesforce/blip-image-captioning-base" \
    --dataset-path ./data/custom_images \
    --num-samples 100 \
    --quantize \
    --output-dir ./results
```

### 3. Compare Results
```python
from benchmark import compare_results

# Compare FP32 vs INT8 performance
comparison = compare_results(
    baseline_path="./results/baseline_results.json",
    quantized_path="./results/quantized_results.json"
)

comparison.plot_comparison()  # Generate comparison plots
comparison.save_report("./results/comparison_report.md")
```

## 📊 Command Line Interface

### Quantization
```bash
# Basic quantization
python quantization.py --model-path "Salesforce/blip-image-captioning-base"

# Advanced options
python quantization.py \
    --model-path "Salesforce/blip-image-captioning-base" \
    --quantization-method int8 \
    --output-path ./models/quantized \
    --validate \
    --num-validation-samples 10
```

### Benchmarking
```bash
# Quick benchmark
python benchmark.py --quick-test

# Full benchmark
python benchmark.py \
    --model-path "Salesforce/blip-image-captioning-base" \
    --annotations ./data/annotations/captions_val2017.json \
    --images ./data/coco/images/val2017 \
    --num-samples 50 \
    --quantize \
    --output benchmark_results.json
```

## 🔍 Sample Outputs

### Quantization Results
```json
{
  "model_info": {
    "original_size_mb": 990.2,
    "quantized_size_mb": 247.8,
    "size_reduction": 0.75,
    "quantization_time": 12.3
  },
  "validation": {
    "bleu_score_original": 0.245,
    "bleu_score_quantized": 0.238,
    "accuracy_retention": 0.971
  }
}
```

### Benchmark Results
```json
{
  "performance_metrics": {
    "avg_inference_time": 1.34,
    "throughput_images_per_sec": 0.746,
    "tokens_per_second": 11.2,
    "memory_usage_mb": 890.5
  },
  "sample_predictions": [
    "a man riding a bike on a city street",
    "a dog sitting on a couch in a living room"
  ]
}
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce batch size or use CPU
python benchmark.py --device cpu --batch-size 1
```

#### 2. Missing Dependencies
```bash
# Install missing packages individually
pip install transformers==4.40.0
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Dataset Download Issues
```python
# Use synthetic data for testing
python benchmark.py --use-synthetic-data --num-samples 10
```

### Verification Steps

#### 1. Test Installation
```python
import torch
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

#### 2. Quick Model Test
```python
from quantization import test_model_loading

# Test if model loads correctly
success = test_model_loading("Salesforce/blip-image-captioning-base")
print(f"Model loading: {'✓' if success else '✗'}")
```

## ⚡ Performance Tips

### 1. Speed Up Processing
- Use `--batch-size 4` for faster inference
- Enable `--use-cache` for repeated runs
- Use `--num-samples 25` for quick testing

### 2. Memory Optimization
- Use `--device cpu` for large models
- Enable `--low-memory-mode` for limited RAM
- Set `--max-memory-gb 8` to limit usage

### 3. Quality vs Speed Trade-off
```bash
# Fast but lower quality
python benchmark.py --num-beams 1 --max-length 20

# Slower but higher quality  
python benchmark.py --num-beams 4 --max-length 50
```

## 📁 Directory Structure After Setup

```
vlm-quantization-benchmark/
├── models/                    # Downloaded/quantized models
│   ├── blip_quantized.pth
│   └── model_info.json
├── data/                      # Dataset files
│   ├── coco/
│   └── synthetic_samples/
├── results/                   # Benchmark outputs
│   ├── baseline_results.json
│   ├── quantized_results.json
│   └── comparison_plots.png
└── logs/                     # Execution logs
    └── benchmark_2025-07-30.log
```

## 🎯 Next Steps

1. **Explore Results**: Check `./results/` for detailed performance metrics
2. **Custom Models**: Try with your own vision-language models
3. **Production Deployment**: Use quantized models in your applications
4. **Advanced Features**: Explore mixed precision and custom quantization schemes

## 💡 Tips for Best Results

- Start with small datasets (25-50 samples) for initial testing
- Use GPU when available for faster benchmarking
- Monitor memory usage with large models
- Save intermediate results to avoid re-computation

Ready to start quantizing? Run your first benchmark:

```bash
python benchmark.py --quick-test
```

🎉 **You're all set!** Check the generated results in `./results/` directory.