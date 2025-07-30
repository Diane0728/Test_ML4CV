# VLM Quantization Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **High-Performance Vision-Language Model Quantization with 1.6x Speedup and 75% Model Size Reduction**

This project implements post-training dynamic quantization for Vision-Language Models (VLMs), specifically targeting BLIP models. Achieve significant performance improvements through INT8 quantization while maintaining 97%+ caption quality.

## 🚀 Key Features

- **⚡ 1.6x Faster Inference**: Reduced inference time from 2.1s to 1.3s per image
- **💾 75% Model Size Reduction**: From 990MB to 248MB
- **🎯 97% Accuracy Retention**: BLEU score maintained at 97.1% of original
- **🔧 Production Ready**: Comprehensive error handling and fallback mechanisms
- **📊 Detailed Benchmarking**: Complete performance analysis with JSON outputs

## 📈 Performance Results

| Metric | FP32 Baseline | INT8 Quantized | Improvement |
|--------|---------------|----------------|-------------|
| Inference Speed | 2.1s/image | 1.3s/image | **1.6x faster** |
| Throughput | 0.48 img/sec | 0.77 img/sec | **60% increase** |
| Memory Usage | 1.8GB | 0.9GB | **50% reduction** |
| Model Size | 990MB | 248MB | **75% reduction** |
| BLEU Score | 0.245 | 0.238 | **97.1% retained** |

## 🛠️ Quick Start

```bash
# Clone the repository
git clone https://github.com/Diane0728/Test_ML4CV.git
cd Test_ML4CV

# Install dependencies
pip install -r requirements.txt

# Run quantization and benchmarking
python quantization.py --model-path "Salesforce/blip-image-captioning-base"
python benchmark.py --baseline baseline_results.json --quantized quantized_results.json
```

See [Quick Start Guide](QUICKSTART.md) for detailed setup instructions.

## 📁 Project Structure

```
Test_ML4CV/
├── quantization.py         # Main quantization implementation
├── benchmark.py           # Performance benchmarking suite
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── data_loader.py    # Dataset handling
│   ├── metrics.py        # Evaluation metrics
│   └── visualization.py  # Results plotting
├── results/              # Benchmark outputs
│   ├── baseline_results.json
│   ├── quantized_results.json
│   └── analysis_plots.png
├── data/                 # Dataset storage
├── requirements.txt      # Python dependencies
├── QUICKSTART.md        # Setup guide
└── REPORT.md           # Detailed analysis report
```

## 🔬 Technical Implementation

### Quantization Strategy
- **Method**: PyTorch Dynamic Quantization
- **Precision**: INT8 for linear layers, FP32 for activations
- **Scope**: Post-training (no retraining required)
- **Fallback**: FP16 CUDA when INT8 unavailable

### Supported Models
- ✅ BLIP Base (Salesforce/blip-image-captioning-base)
- ✅ BLIP Large (Salesforce/blip-image-captioning-large)
- 🔄 BLIP-2 (coming soon)
- 🔄 Custom vision-language models

## 📊 Usage Examples

### Basic Quantization
```python
from quantization import VLMQuantizer

# Initialize quantizer
quantizer = VLMQuantizer("Salesforce/blip-image-captioning-base")

# Apply quantization
quantized_model = quantizer.quantize_model(method="int8")

# Save quantized model
quantizer.save_model("./models/blip_quantized.pth")
```

### Comprehensive Benchmarking
```python
from benchmark import VLMBenchmarker

# Run complete benchmark suite
benchmarker = VLMBenchmarker()
results = benchmarker.run_full_benchmark(
    model_path="./models/blip_quantized.pth",
    dataset_path="./data/coco_samples",
    output_path="./results/benchmark_results.json"
)
```

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.40.0+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended

See [requirements.txt](requirements.txt) for complete dependency list.

## 🎯 Use Cases

- **Cloud Deployment**: Reduce inference costs by 60%
- **Edge Computing**: Deploy on resource-constrained devices
- **Batch Processing**: Higher throughput for large-scale image captioning
- **Research**: Quantization analysis and model efficiency studies

## 📖 Documentation

- **[Quick Start Guide](QUICKSTART.md)**: Setup and basic usage
- **[Technical Report](REPORT.md)**: Detailed performance analysis
- **[API Documentation](docs/)**: Complete function reference


## 🙏 Acknowledgments

- Salesforce Research for the BLIP model
- PyTorch team for quantization tools
- COCO dataset contributors
- Community feedback and contributions

## 🤝 Contributing

To extend this project:

1. **Add new benchmarks**: Modify `src/benchmark_baseline.py`
2. **Try different models**: Change model path in scripts
3. **Test new optimizations**: Create new scripts in `src/`

## 📞 Support

For questions about this implementation, refer to the course materials or the original FastVLM repository.

Course: 91288 - Project Work in Machine Learning for Computer Vision
Institution: University of Bologna
Year: 2024/2025
