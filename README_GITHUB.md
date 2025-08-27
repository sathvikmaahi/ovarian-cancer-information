# Multi-Modal Cancer Classification

A comprehensive multi-modal classification pipeline that combines image and tabular features for improved cancer diagnosis using synthetic data.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python run_pipeline.py
```

## 🏗️ Architecture

- **Image-Only CNN**: 3-layer CNN for image classification
- **Tabular-Only MLP**: Multi-layer perceptron for tabular features  
- **Fused Multi-Modal**: Combines both modalities with fusion layers

## 📁 Core Files

- `src/data_preparation.py` - Synthetic data generation
- `src/model_implementation.py` - Model training and evaluation
- `run_pipeline.py` - Main execution script
- `requirements.txt` - Python dependencies

## 📊 Features

- Synthetic biomarker generation (Age, CA-125, BRCA status)
- 64x64 grayscale synthetic images
- Breast Cancer Wisconsin dataset integration
- Comprehensive model evaluation
- Early stopping and learning rate scheduling

## ⚠️ Note

This project uses synthetic data for demonstration purposes only.
