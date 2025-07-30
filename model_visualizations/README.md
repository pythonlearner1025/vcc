# Model Architecture Visualizations

This directory contains comprehensive visualizations of the Conditional Diffusion Transformer model.

## Generated Files

### Configuration Files
- **`model_config.json`**: Complete model configuration in JSON format
- **`model_config.txt`**: Human-readable configuration summary

### Architecture Diagrams
- **`architecture_overview.png`**: High-level model architecture overview
- **`layer_breakdown.png`**: Detailed breakdown of each layer type
- **`forward_pass_flow.png`**: Step-by-step forward pass visualization
- **`attention_mechanism.png`**: Multi-Query Attention and Cross-Attention details
- **`conditioning_pipeline.png`**: Perturbation conditioning pathway
- **`diffusion_process.png`**: Forward and reverse diffusion processes

### Analysis Files
- **`parameter_breakdown.png`**: Parameter distribution across components  
- **`parameter_breakdown.txt`**: Detailed parameter counts
- **`computational_analysis.png`**: FLOP distribution and scaling analysis
- **`computational_analysis.txt`**: Computational requirements breakdown

## Model Overview

**Architecture**: Conditional Diffusion Transformer
**Parameters**: 24,752,265
**Model Size**: ~94.4 MB (FP32)

### Key Components:
1. **Token Embeddings**: Maps expression bins to 512D vectors
2. **Conditioning**: Perturbation (gene, sign, magnitude) + control set encoding
3. **Transformer Stack**: 6 layers with Multi-Query Attention
4. **Diffusion Process**: 10 timesteps with 0.3 masking
5. **Output Head**: Predicts original tokens for masked positions

### Conditioning Types:
- **Target Gene**: Which gene is perturbed (36,601 gene vocabulary)
- **Perturbation Sign**: Direction (-1: knockdown, 0: control, +1: activation)  
- **Perturbation Magnitude**: Continuous log2 fold change value
- **Control Set**: Reference cells for comparison (encoded via set transformer)

### Training Strategy:
- **Pretraining**: 50 epochs on diverse scRNA-seq data
- **Fine-tuning**: 20 epochs on perturbation data
- **Diffusion**: Partial masking with curriculum learning
- **Loss**: Cross-entropy on masked positions

## Usage Notes

1. **Input Format**: Tokenized gene expression (64 bins)
2. **Output Format**: Logits over vocabulary for each gene position
3. **Inference**: Iterative denoising from fully masked input
4. **Conditioning**: Optional - model can run unconditionally

## Implementation Details

- **Attention**: Multi-Query Attention for efficiency
- **Memory**: Scales linearly with batch size, quadratically with gene count
- **Precision**: Supports FP16 training for memory efficiency
- **Hardware**: Requires ~0.1 GB GPU memory for training

Generated on: 2025-07-29
Model Configuration: Conditional Diffusion Transformer v1.0
