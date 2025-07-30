#!/usr/bin/env python3
"""
Model Architecture Visualization Script

Creates comprehensive visualizations of the conditional diffusion transformer model
without loading it into memory. Generates architecture diagrams, forward pass flows,
and detailed configuration information.

Usage:
    python scripts/visualize_model.py --output-dir visualizations/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import seaborn as sns
from dataclasses import asdict

# Import model config without instantiating the model
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.diffusion import ConditionalModelConfig


class ModelVisualizer:
    """Visualize model architecture and forward pass without instantiation."""
    
    def __init__(self, config: ConditionalModelConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_all_visualizations(self):
        """Generate all visualizations and save to output directory."""
        print(f"Creating model visualizations in {self.output_dir}")
        
        # 1. Model configuration summary
        self.save_config_info()
        
        # 2. Architecture overview diagram
        self.create_architecture_diagram()
        
        # 3. Detailed layer breakdown
        self.create_layer_breakdown()
        
        # 4. Forward pass flow diagram
        self.create_forward_pass_diagram()
        
        # 5. Attention mechanism visualization
        self.create_attention_diagram()
        
        # 6. Conditioning pipeline diagram
        self.create_conditioning_diagram()
        
        # 7. Diffusion process visualization
        self.create_diffusion_diagram()
        
        # 8. Parameter count breakdown
        self.create_parameter_breakdown()
        
        # 9. Model size and computational cost analysis
        self.create_computational_analysis()
        
        print(f"All visualizations saved to {self.output_dir}")
        print(f"Open {self.output_dir}/README.md for a guide to the generated files")
        
        # Generate README for the visualizations
        self.create_visualization_readme()
    
    def save_config_info(self):
        """Save detailed configuration information."""
        config_dict = asdict(self.config)
        
        # Save as JSON
        with open(self.output_dir / "model_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save as human-readable text
        with open(self.output_dir / "model_config.txt", 'w') as f:
            f.write("Conditional Diffusion Transformer Configuration\n")
            f.write("=" * 50 + "\n\n")
            
            # Architecture section
            f.write("ARCHITECTURE:\n")
            f.write(f"  Model Dimension: {self.config.dim}\n")
            f.write(f"  Number of Heads: {self.config.n_head}\n")
            f.write(f"  Number of Layers: {self.config.n_layer}\n")
            f.write(f"  FFN Multiplier: {self.config.ffn_mult}\n")
            f.write(f"  Vocabulary Size: {self.config.vocab_size}\n")
            f.write(f"  Number of HVGs: {self.config.n_genes}\n")
            f.write(f"  Total Genes in Vocab: {self.config.n_total_genes}\n\n")
            
            # Conditioning section
            f.write("CONDITIONING:\n")
            f.write(f"  Gene Embedding Dim: {self.config.gene_embed_dim}\n")
            f.write(f"  Perturbation Sign Dim: {self.config.perturb_sign_dim}\n")
            f.write(f"  Perturbation Magnitude Dim: {self.config.perturb_magnitude_dim}\n")
            f.write(f"  Magnitude Clipping: ±{self.config.magnitude_clip}\n")
            f.write(f"  Control Set Encoder Layers: {self.config.control_set_encoder_layers}\n")
            f.write(f"  Control Set Hidden Dim: {self.config.control_set_dim_hidden}\n")
            f.write(f"  Control Set Output Dim: {self.config.control_set_dim_out}\n\n")
            
            # Diffusion section
            f.write("DIFFUSION:\n")
            f.write(f"  Number of Timesteps: {self.config.n_timesteps}\n")
            f.write(f"  Mask Ratio: {self.config.mask_ratio}\n")
            f.write(f"  Schedule: {self.config.schedule}\n\n")
            
            # Training section
            f.write("TRAINING:\n")
            f.write(f"  Batch Size: {self.config.batch_size}\n")
            f.write(f"  Learning Rate: {self.config.learning_rate}\n")
            f.write(f"  Weight Decay: {self.config.weight_decay}\n")
            f.write(f"  Warmup Steps: {self.config.warmup_steps}\n")
            f.write(f"  Pretrain Epochs: {self.config.pretrain_epochs}\n")
            f.write(f"  Finetune Epochs: {self.config.finetune_epochs}\n\n")
            
            # Parameters section
            f.write("MODEL SIZE:\n")
            f.write(f"  Estimated Parameters: {self.config.n_params:,}\n")
            f.write(f"  Model Size (approx): {self.config.n_params * 4 / 1024**2:.1f} MB\n")
    
    def create_architecture_diagram(self):
        """Create high-level architecture overview."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Define colors
        colors = {
            'input': '#E8F4FD',
            'embedding': '#B8E6B8', 
            'conditioning': '#FFE4B5',
            'transformer': '#DDA0DD',
            'output': '#FFA07A',
            'diffusion': '#F0E68C'
        }
        
        # Input layer
        input_box = FancyBboxPatch((0.5, 10.5), 2, 1, boxstyle="round,pad=0.1", 
                                   facecolor=colors['input'], edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(1.5, 11, 'Input Tokens\n(B, N)', ha='center', va='center', fontsize=10, weight='bold')
        
        # Embedding layers
        token_emb = FancyBboxPatch((0.5, 9), 2, 0.8, boxstyle="round,pad=0.1",
                                   facecolor=colors['embedding'], edgecolor='black')
        ax.add_patch(token_emb)
        ax.text(1.5, 9.4, 'Token Embedding\n+ Position Embedding', ha='center', va='center', fontsize=9)
        
        pos_emb = FancyBboxPatch((3, 9), 2, 0.8, boxstyle="round,pad=0.1",
                                 facecolor=colors['embedding'], edgecolor='black')
        ax.add_patch(pos_emb)
        ax.text(4, 9.4, 'Time Embedding\n(Sinusoidal)', ha='center', va='center', fontsize=9)
        
        # Conditioning pathway
        cond_box = FancyBboxPatch((6, 8.5), 3, 2.5, boxstyle="round,pad=0.1",
                                  facecolor=colors['conditioning'], edgecolor='black', linewidth=2)
        ax.add_patch(cond_box)
        ax.text(7.5, 10.5, 'Conditioning Pipeline', ha='center', va='center', fontsize=11, weight='bold')
        ax.text(7.5, 10, '• Gene Embedding', ha='center', va='center', fontsize=9)
        ax.text(7.5, 9.6, '• Perturbation Sign', ha='center', va='center', fontsize=9)
        ax.text(7.5, 9.2, '• Perturbation Magnitude', ha='center', va='center', fontsize=9)
        ax.text(7.5, 8.8, '• Control Set Encoder', ha='center', va='center', fontsize=9)
        
        # Transformer stack
        for i in range(self.config.n_layer):
            y_pos = 7.5 - i * 0.8
            layer_box = FancyBboxPatch((1, y_pos), 4, 0.6, boxstyle="round,pad=0.05",
                                       facecolor=colors['transformer'], edgecolor='black')
            ax.add_patch(layer_box)
            ax.text(3, y_pos + 0.3, f'Transformer Block {i+1}\n(Self-Attn + Cross-Attn + FFN)', 
                   ha='center', va='center', fontsize=8)
        
        # Output layer
        output_box = FancyBboxPatch((1.5, 0.5), 3, 1, boxstyle="round,pad=0.1",
                                    facecolor=colors['output'], edgecolor='black', linewidth=2)
        ax.add_patch(output_box)
        ax.text(3, 1, 'Output Head\n(Vocab Logits)', ha='center', va='center', fontsize=10, weight='bold')
        
        # Diffusion process annotation
        diff_box = FancyBboxPatch((6, 0.5), 3, 2, boxstyle="round,pad=0.1",
                                  facecolor=colors['diffusion'], edgecolor='black', linewidth=2)
        ax.add_patch(diff_box)
        ax.text(7.5, 1.8, 'Diffusion Process', ha='center', va='center', fontsize=11, weight='bold')
        ax.text(7.5, 1.4, f'{self.config.n_timesteps} timesteps', ha='center', va='center', fontsize=9)
        ax.text(7.5, 1.1, f'{self.config.mask_ratio} mask ratio', ha='center', va='center', fontsize=9)
        ax.text(7.5, 0.8, f'{self.config.schedule} schedule', ha='center', va='center', fontsize=9)
        
        # Add arrows
        arrows = [
            ((1.5, 10.5), (1.5, 9.8)),  # Input to token embedding
            ((1.5, 9), (1.5, 8.1)),     # Token embedding to transformer
            ((4, 9), (3, 8.1)),         # Time embedding to transformer
            ((6, 9.75), (5, 8.1)),      # Conditioning to transformer
            ((3, 2.3), (3, 1.5)),       # Last transformer to output
        ]
        
        for (x1, y1), (x2, y2) in arrows:
            arrow = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                                   arrowstyle="->", shrinkA=5, shrinkB=5, 
                                   mutation_scale=20, fc="black")
            ax.add_patch(arrow)
        
        ax.set_title('Conditional Diffusion Transformer Architecture', fontsize=16, weight='bold', pad=20)
        
        # Add parameter info
        ax.text(5, -0.5, f'Total Parameters: {self.config.n_params:,}', 
               ha='center', va='center', fontsize=12, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_layer_breakdown(self):
        """Create detailed breakdown of each layer type."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Multi-Query Attention
        ax = axes[0, 0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Multi-Query Attention (MQA)', fontsize=14, weight='bold')
        
        # Input
        input_rect = patches.Rectangle((1, 8), 8, 1, linewidth=2, edgecolor='black', facecolor='lightblue')
        ax.add_patch(input_rect)
        ax.text(5, 8.5, f'Input: (B, N, {self.config.dim})', ha='center', va='center', fontsize=10)
        
        # Q, K, V projections
        q_rect = patches.Rectangle((1, 6.5), 2, 1, linewidth=1, edgecolor='black', facecolor='lightgreen')
        ax.add_patch(q_rect)
        ax.text(2, 7, f'Q Proj\n{self.config.n_head} heads', ha='center', va='center', fontsize=9)
        
        kv_rect = patches.Rectangle((4, 6.5), 2, 1, linewidth=1, edgecolor='black', facecolor='lightcoral')
        ax.add_patch(kv_rect)
        ax.text(5, 7, 'Shared K,V\n1 head', ha='center', va='center', fontsize=9)
        
        # Attention
        attn_rect = patches.Rectangle((2, 4.5), 4, 1, linewidth=2, edgecolor='black', facecolor='lightyellow')
        ax.add_patch(attn_rect)
        ax.text(4, 5, 'Scaled Dot-Product\nAttention', ha='center', va='center', fontsize=10)
        
        # Output
        out_rect = patches.Rectangle((2, 2.5), 4, 1, linewidth=2, edgecolor='black', facecolor='lightpink')
        ax.add_patch(out_rect)
        ax.text(4, 3, f'Output Proj\n(B, N, {self.config.dim})', ha='center', va='center', fontsize=10)
        
        # Transformer Block
        ax = axes[0, 1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        ax.set_title('Transformer Block', fontsize=14, weight='bold')
        
        # Layer components
        components = [
            ('LayerNorm', 10.5, 'lightblue'),
            ('Self-Attention', 9.5, 'lightgreen'),
            ('Residual Add', 8.5, 'lightyellow'),
            ('LayerNorm', 7.5, 'lightblue'),
            ('Cross-Attention', 6.5, 'lightcoral'),
            ('Residual Add', 5.5, 'lightyellow'),
            ('LayerNorm', 4.5, 'lightblue'),
            ('FFN (4x expansion)', 3.5, 'lightpink'),
            ('Residual Add', 2.5, 'lightyellow'),
        ]
        
        for comp, y, color in components:
            rect = patches.Rectangle((2, y-0.4), 6, 0.8, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(5, y, comp, ha='center', va='center', fontsize=9)
        
        # Control Set Encoder
        ax = axes[1, 0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Control Set Encoder', fontsize=14, weight='bold')
        
        # Components
        ctrl_components = [
            (f'Input: (B, S, {self.config.n_genes})', 8.5, 'lightblue'),
            (f'Cell MLP: → {self.config.control_set_dim_hidden}', 7, 'lightgreen'),
            (f'Set Transformer\n{self.config.control_set_encoder_layers} layers', 5.5, 'lightyellow'),
            (f'Output Proj: → {self.config.control_set_dim_out}', 4, 'lightpink'),
            (f'Output: (B, S, {self.config.control_set_dim_out})', 2.5, 'lightcoral'),
        ]
        
        for comp, y, color in ctrl_components:
            rect = patches.Rectangle((1, y-0.4), 8, 0.8, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(5, y, comp, ha='center', va='center', fontsize=9)
        
        # Conditioning Pipeline
        ax = axes[1, 1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        ax.set_title('Conditioning Pipeline', fontsize=14, weight='bold')
        
        # Conditioning components
        cond_components = [
            (f'Gene Embedding\n{self.config.n_total_genes} → {self.config.gene_embed_dim}', 10.5, 'lightblue'),
            (f'Sign Embedding\n3 → {self.config.perturb_sign_dim}', 9, 'lightgreen'),
            (f'Magnitude MLP\n1 → {self.config.perturb_magnitude_dim} → {self.config.dim}', 7.5, 'lightyellow'),
            (f'Concatenate\n{self.config.gene_embed_dim + self.config.perturb_sign_dim + self.config.dim}', 6, 'lightcoral'),
            (f'Conditioning Proj\n→ {self.config.dim}', 4.5, 'lightpink'),
            (f'Add to Time Emb\n(B, {self.config.dim})', 3, 'lavender'),
        ]
        
        for comp, y, color in cond_components:
            rect = patches.Rectangle((1, y-0.4), 8, 0.8, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(5, y, comp, ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_forward_pass_diagram(self):
        """Create step-by-step forward pass visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 20)
        ax.axis('off')
        
        # Forward pass steps
        steps = [
            ('Input Tokens', f'(B={self.config.batch_size}, N={self.config.n_genes})', 18.5, 'lightblue'),
            ('Token Embedding', f'→ (B, N, {self.config.dim})', 17.5, 'lightgreen'),
            ('+ Position Embedding', f'(1, N, {self.config.dim})', 16.5, 'lightgreen'),
            ('Time Embedding', f'(B,) → (B, {self.config.dim})', 15.5, 'lightyellow'),
            ('Conditioning', '', 14.5, 'lightcoral'),
            ('  • Gene Embed', f'{self.config.gene_embed_dim}D', 14, 'lightcoral'),
            ('  • Sign Embed', f'{self.config.perturb_sign_dim}D', 13.5, 'lightcoral'),
            ('  • Magnitude MLP', f'{self.config.dim}D', 13, 'lightcoral'),
            ('  • Combine & Project', f'→ {self.config.dim}D', 12.5, 'lightcoral'),
            ('Control Set Encoding', f'(B, S, N) → (B, S, {self.config.control_set_dim_out})', 11.5, 'lavender'),
            ('Add Time + Conditioning', f'(B, N, {self.config.dim})', 10.5, 'wheat'),
        ]
        
        # Transformer layers
        for i in range(self.config.n_layer):
            y_pos = 9.5 - i * 1.2
            steps.extend([
                (f'Layer {i+1}: Self-Attention', f'(B, N, {self.config.dim})', y_pos, 'lightpink'),
                (f'Layer {i+1}: Cross-Attention', f'with control context', y_pos - 0.3, 'lightpink'),
                (f'Layer {i+1}: FFN', f'{self.config.ffn_mult}x expansion', y_pos - 0.6, 'lightpink'),
            ])
        
        final_y = 9.5 - self.config.n_layer * 1.2
        steps.extend([
            ('Final LayerNorm', f'(B, N, {self.config.dim})', final_y - 1, 'lightsteelblue'),
            ('Output Head', f'→ (B, N, {self.config.vocab_size})', final_y - 1.5, 'lightsalmon'),
            ('Logits', f'Vocabulary probabilities', final_y - 2, 'gold'),
        ])
        
        # Draw steps
        for step, shape, y, color in steps:
            if step.startswith('  •'):
                # Indent sub-steps
                rect = patches.Rectangle((2, y-0.15), 6, 0.3, linewidth=1, edgecolor='gray', facecolor=color)
                ax.text(1.8, y, '•', ha='center', va='center', fontsize=10)
                ax.text(3, y, step[3:], ha='left', va='center', fontsize=9)
                ax.text(8.5, y, shape, ha='center', va='center', fontsize=8, style='italic')
            else:
                rect = patches.Rectangle((1, y-0.2), 8, 0.4, linewidth=1, edgecolor='black', facecolor=color)
                ax.text(2, y, step, ha='left', va='center', fontsize=10, weight='bold' if 'Layer' not in step else 'normal')
                ax.text(8.5, y, shape, ha='center', va='center', fontsize=9, style='italic')
            
            ax.add_patch(rect)
            
            # Add arrows between major steps
            if y > final_y - 2 and not step.startswith('  •') and 'Layer' not in step:
                arrow = patches.FancyArrowPatch((5, y-0.2), (5, y-0.6),
                                               arrowstyle='->', mutation_scale=15, color='black')
                ax.add_patch(arrow)
        
        ax.set_title('Forward Pass Flow', fontsize=16, weight='bold', pad=20)
        
        # Add tensor shape legend
        legend_text = (
            "Tensor Shape Legend:\n"
            f"B = Batch size ({self.config.batch_size})\n"
            f"N = Number of genes ({self.config.n_genes})\n"
            f"S = Control set size\n"
            f"D = Model dimension ({self.config.dim})"
        )
        ax.text(0.5, 1.5, legend_text, ha='left', va='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'forward_pass_flow.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_attention_diagram(self):
        """Visualize attention mechanism in detail."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Self-attention
        ax = axes[0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Multi-Query Self-Attention', fontsize=14, weight='bold')
        
        # Input
        input_rect = patches.Rectangle((1, 8.5), 8, 0.8, linewidth=2, edgecolor='black', facecolor='lightblue')
        ax.add_patch(input_rect)
        ax.text(5, 8.9, f'Input: (B, {self.config.n_genes}, {self.config.dim})', ha='center', va='center', fontsize=10)
        
        # Q projection (all heads)
        q_rect = patches.Rectangle((0.5, 7), 3, 0.8, linewidth=1, edgecolor='black', facecolor='lightgreen')
        ax.add_patch(q_rect)
        ax.text(2, 7.4, f'Q: {self.config.n_head} heads\n({self.config.dim//self.config.n_head} each)', ha='center', va='center', fontsize=9)
        
        # Shared K,V projection
        kv_rect = patches.Rectangle((6.5, 7), 2.5, 0.8, linewidth=1, edgecolor='black', facecolor='lightcoral')
        ax.add_patch(kv_rect)
        ax.text(7.75, 7.4, f'Shared K,V\n1 head', ha='center', va='center', fontsize=9)
        
        # Attention computation
        attn_rect = patches.Rectangle((2, 5.5), 6, 0.8, linewidth=2, edgecolor='black', facecolor='lightyellow')
        ax.add_patch(attn_rect)
        ax.text(5, 5.9, f'Attention: QK^T / √{self.config.dim//self.config.n_head}', ha='center', va='center', fontsize=10)
        
        # Output
        out_rect = patches.Rectangle((2.5, 4), 5, 0.8, linewidth=2, edgecolor='black', facecolor='lightpink')
        ax.add_patch(out_rect)
        ax.text(5, 4.4, f'Output: (B, {self.config.n_genes}, {self.config.dim})', ha='center', va='center', fontsize=10)
        
        # Cross-attention
        ax = axes[1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Cross-Attention with Control Set', fontsize=14, weight='bold')
        
        # Sequence input (Q)
        seq_rect = patches.Rectangle((1, 8.5), 3.5, 0.8, linewidth=2, edgecolor='black', facecolor='lightblue')
        ax.add_patch(seq_rect)
        ax.text(2.75, 8.9, f'Sequence (Q)\n(B, {self.config.n_genes}, {self.config.dim})', ha='center', va='center', fontsize=9)
        
        # Control context (K,V)
        ctx_rect = patches.Rectangle((5.5, 8.5), 3.5, 0.8, linewidth=2, edgecolor='black', facecolor='lavender')
        ax.add_patch(ctx_rect)
        ax.text(7.25, 8.9, f'Control Set (K,V)\n(B, S, {self.config.control_set_dim_out})', ha='center', va='center', fontsize=9)
        
        # Cross-attention
        cross_attn_rect = patches.Rectangle((2, 6.5), 6, 0.8, linewidth=2, edgecolor='black', facecolor='lightyellow')
        ax.add_patch(cross_attn_rect)
        ax.text(5, 6.9, 'Cross-Attention: Q(seq) × K,V(control)', ha='center', va='center', fontsize=10)
        
        # Output
        cross_out_rect = patches.Rectangle((2.5, 5), 5, 0.8, linewidth=2, edgecolor='black', facecolor='lightpink')
        ax.add_patch(cross_out_rect)
        ax.text(5, 5.4, f'Attended Sequence\n(B, {self.config.n_genes}, {self.config.dim})', ha='center', va='center', fontsize=10)
        
        # Attention pattern visualization
        attention_rect = patches.Rectangle((1, 2), 8, 2, linewidth=1, edgecolor='black', facecolor='white')
        ax.add_patch(attention_rect)
        
        # Simple attention heatmap
        np.random.seed(42)
        attention_matrix = np.random.rand(8, 12)
        attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
        
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto', extent=[1.2, 8.8, 2.2, 3.8])
        ax.text(5, 1.5, 'Example Attention Pattern\n(Genes × Control Cells)', ha='center', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_mechanism.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_conditioning_diagram(self):
        """Visualize the conditioning pipeline in detail."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Gene conditioning branch
        gene_input = patches.Rectangle((1, 10), 2, 1, linewidth=2, edgecolor='black', facecolor='lightblue')
        ax.add_patch(gene_input)
        ax.text(2, 10.5, 'Target Gene\nIndex', ha='center', va='center', fontsize=10, weight='bold')
        
        gene_embed = patches.Rectangle((1, 8.5), 2, 1, linewidth=1, edgecolor='black', facecolor='lightgreen')
        ax.add_patch(gene_embed)
        ax.text(2, 9, f'Gene Embedding\n{self.config.n_total_genes}→{self.config.gene_embed_dim}', ha='center', va='center', fontsize=9)
        
        # Sign conditioning branch
        sign_input = patches.Rectangle((4, 10), 2, 1, linewidth=2, edgecolor='black', facecolor='lightblue')
        ax.add_patch(sign_input)
        ax.text(5, 10.5, 'Perturbation\nSign (-1,0,+1)', ha='center', va='center', fontsize=10, weight='bold')
        
        sign_embed = patches.Rectangle((4, 8.5), 2, 1, linewidth=1, edgecolor='black', facecolor='lightcoral')
        ax.add_patch(sign_embed)
        ax.text(5, 9, f'Sign Embedding\n3→{self.config.perturb_sign_dim}', ha='center', va='center', fontsize=9)
        
        # Magnitude conditioning branch
        mag_input = patches.Rectangle((7, 10), 2, 1, linewidth=2, edgecolor='black', facecolor='lightblue')
        ax.add_patch(mag_input)
        ax.text(8, 10.5, 'Log2 Fold\nChange', ha='center', va='center', fontsize=10, weight='bold')
        
        mag_mlp = patches.Rectangle((7, 8.5), 2, 1, linewidth=1, edgecolor='black', facecolor='lightyellow')
        ax.add_patch(mag_mlp)
        ax.text(8, 9, f'Magnitude MLP\n1→{self.config.perturb_magnitude_dim}→{self.config.dim}', ha='center', va='center', fontsize=9)
        
        # Control set branch
        ctrl_input = patches.Rectangle((10.5, 10), 2.5, 1, linewidth=2, edgecolor='black', facecolor='lightblue')
        ax.add_patch(ctrl_input)
        ax.text(11.75, 10.5, 'Control Set\n(B, S, N)', ha='center', va='center', fontsize=10, weight='bold')
        
        ctrl_encoder = patches.Rectangle((10.5, 8.5), 2.5, 1, linewidth=1, edgecolor='black', facecolor='lavender')
        ax.add_patch(ctrl_encoder)
        ax.text(11.75, 9, f'Set Encoder\n→(B,S,{self.config.control_set_dim_out})', ha='center', va='center', fontsize=9)
        
        # Concatenation
        concat_box = patches.Rectangle((2, 6.5), 6, 1, linewidth=2, edgecolor='black', facecolor='wheat')
        ax.add_patch(concat_box)
        ax.text(5, 7, f'Concatenate\n{self.config.gene_embed_dim}+{self.config.perturb_sign_dim}+{self.config.dim} = {self.config.gene_embed_dim + self.config.perturb_sign_dim + self.config.dim}', ha='center', va='center', fontsize=10)
        
        # Conditioning projection
        cond_proj = patches.Rectangle((3, 5), 4, 1, linewidth=2, edgecolor='black', facecolor='lightpink')
        ax.add_patch(cond_proj)
        ax.text(5, 5.5, f'Conditioning Projection\n→ {self.config.dim}D', ha='center', va='center', fontsize=10)
        
        # Time embedding
        time_box = patches.Rectangle((1, 3.5), 3, 1, linewidth=2, edgecolor='black', facecolor='lightsteelblue')
        ax.add_patch(time_box)
        ax.text(2.5, 4, f'Time Embedding\n{self.config.dim}D', ha='center', va='center', fontsize=10)
        
        # Final combination
        final_box = patches.Rectangle((3, 2), 4, 1, linewidth=3, edgecolor='black', facecolor='gold')
        ax.add_patch(final_box)
        ax.text(5, 2.5, f'Time + Conditioning\n(B, {self.config.dim})', ha='center', va='center', fontsize=11, weight='bold')
        
        # Control context (separate path)
        ctx_box = patches.Rectangle((9, 2), 4, 1, linewidth=3, edgecolor='black', facecolor='mediumpurple')
        ax.add_patch(ctx_box)
        ax.text(11, 2.5, f'Control Context\n(B, S, {self.config.control_set_dim_out})', ha='center', va='center', fontsize=11, weight='bold')
        
        # Add arrows
        arrows = [
            # Vertical arrows from inputs to embeddings
            ((2, 10), (2, 9.5)),
            ((5, 10), (5, 9.5)),
            ((8, 10), (8, 9.5)),
            ((11.75, 10), (11.75, 9.5)),
            # From embeddings to concatenation
            ((2, 8.5), (3, 7.5)),
            ((5, 8.5), (5, 7.5)),
            ((8, 8.5), (7, 7.5)),
            # From concat to projection
            ((5, 6.5), (5, 6)),
            # From projection and time to final
            ((5, 5), (5, 3)),
            ((2.5, 3.5), (4, 3)),
            # Control set to context
            ((11.75, 8.5), (11, 3)),
        ]
        
        for (x1, y1), (x2, y2) in arrows:
            arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                           arrowstyle='->', mutation_scale=15, color='black')
            ax.add_patch(arrow)
        
        ax.set_title('Conditioning Pipeline', fontsize=16, weight='bold', pad=20)
        
        # Add explanatory text
        ax.text(7, 0.5, 'Two conditioning paths: (1) Combined time+perturbation added to sequence, (2) Control set used for cross-attention',
               ha='center', va='center', fontsize=10, style='italic',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'conditioning_pipeline.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_diffusion_diagram(self):
        """Visualize the diffusion process."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Forward process (noise addition)
        ax = axes[0]
        ax.set_title('Forward Diffusion Process (Training)', fontsize=14, weight='bold')
        
        # Create example data for visualization
        n_steps = 5  # Show 5 steps for clarity
        gene_example = np.array([15, 0, 8, 0, 23, 0, 12, 4, 0, 19])  # Example gene expression
        
        for i in range(n_steps):
            # Simulate increasing mask ratio
            mask_ratio = i / (n_steps - 1) * self.config.mask_ratio
            masked_genes = gene_example.copy()
            n_mask = int(len(gene_example) * mask_ratio)
            mask_indices = np.random.choice(len(gene_example), n_mask, replace=False)
            masked_genes[mask_indices] = 63  # Mask token
            
            # Plot
            x_pos = i * 2.5 + 1
            ax.bar(np.arange(len(gene_example)) * 0.2 + x_pos, masked_genes, width=0.15, 
                   color=['red' if x == 63 else 'blue' for x in masked_genes])
            ax.text(x_pos + 1, max(gene_example) + 5, f't={i}\n{mask_ratio:.1%} masked', 
                   ha='center', fontsize=10)
            
            if i < n_steps - 1:
                ax.arrow(x_pos + 2.2, max(gene_example) / 2, 0.5, 0, head_width=2, head_length=0.1, fc='black')
        
        ax.set_xlim(0, n_steps * 2.5)
        ax.set_ylim(0, max(gene_example) + 15)
        ax.set_ylabel('Gene Expression')
        ax.set_xlabel('Diffusion Steps')
        
        # Reverse process (denoising)
        ax = axes[1]
        ax.set_title('Reverse Diffusion Process (Generation)', fontsize=14, weight='bold')
        
        for i in range(n_steps):
            # Simulate decreasing mask ratio (reverse process)
            mask_ratio = (n_steps - 1 - i) / (n_steps - 1) * self.config.mask_ratio
            unmasked_genes = gene_example.copy()
            n_mask = int(len(gene_example) * mask_ratio)
            if n_mask > 0:
                mask_indices = np.random.choice(len(gene_example), n_mask, replace=False)
                unmasked_genes[mask_indices] = 63  # Still masked
            
            # Plot
            x_pos = i * 2.5 + 1
            ax.bar(np.arange(len(gene_example)) * 0.2 + x_pos, unmasked_genes, width=0.15,
                   color=['red' if x == 63 else 'green' for x in unmasked_genes])
            ax.text(x_pos + 1, max(gene_example) + 5, f't={n_steps-1-i}\n{mask_ratio:.1%} masked', 
                   ha='center', fontsize=10)
            
            if i < n_steps - 1:
                ax.arrow(x_pos + 2.2, max(gene_example) / 2, 0.5, 0, head_width=2, head_length=0.1, fc='black')
        
        ax.set_xlim(0, n_steps * 2.5)
        ax.set_ylim(0, max(gene_example) + 15)
        ax.set_ylabel('Gene Expression')
        ax.set_xlabel('Denoising Steps')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='Original Token'),
                          Patch(facecolor='red', label='[MASK] Token'),
                          Patch(facecolor='green', label='Generated Token')]
        axes[1].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'diffusion_process.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_parameter_breakdown(self):
        """Create parameter count breakdown chart."""
        # Calculate parameter counts for each component
        components = {
            'Token Embedding': self.config.vocab_size * self.config.dim,
            'Position Embedding': self.config.n_genes * self.config.dim,
            'Time Embedding': self.config.dim * 2,  # Two linear layers
            'Gene Embedding': self.config.n_total_genes * self.config.gene_embed_dim,
            'Sign Embedding': 3 * self.config.perturb_sign_dim,
            'Magnitude MLP': self.config.perturb_magnitude_dim * self.config.dim * 3,  # 3 layers
            'Conditioning Proj': (self.config.gene_embed_dim + self.config.perturb_sign_dim + self.config.dim) * self.config.dim * 2,
            'Control Set Encoder': self.config.n_genes * self.config.control_set_dim_hidden * 3,  # Simplified estimate
            'Transformer Layers': self.config.n_layer * (4 * self.config.dim * self.config.dim + 2 * self.config.dim * self.config.dim * self.config.ffn_mult),
            'Output Head': self.config.dim * self.config.vocab_size,
        }
        
        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Pie chart
        labels = list(components.keys())
        sizes = list(components.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Parameter Distribution\nTotal: {sum(sizes):,} parameters', fontsize=14, weight='bold')
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        # Bar chart with exact numbers
        ax2.barh(labels, [x/1e6 for x in sizes], color=colors)
        ax2.set_xlabel('Parameters (Millions)')
        ax2.set_title('Parameter Count by Component', fontsize=14, weight='bold')
        
        # Add value labels on bars
        for i, v in enumerate(sizes):
            ax2.text(v/1e6 + 0.1, i, f'{v/1e6:.1f}M', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed breakdown as text
        with open(self.output_dir / 'parameter_breakdown.txt', 'w') as f:
            f.write("Detailed Parameter Breakdown\n")
            f.write("=" * 40 + "\n\n")
            
            total = sum(sizes)
            for component, count in components.items():
                percentage = count / total * 100
                f.write(f"{component:<25}: {count:>10,} ({percentage:>5.1f}%)\n")
            
            f.write("\n" + "-" * 40 + "\n")
            f.write(f"{'TOTAL':<25}: {total:>10,} (100.0%)\n")
            f.write(f"\nModel size (FP32): {total * 4 / 1024**2:.1f} MB\n")
            f.write(f"Model size (FP16): {total * 2 / 1024**2:.1f} MB\n")
    
    def create_computational_analysis(self):
        """Analyze computational requirements."""
        # Calculate FLOPs for different operations
        B = self.config.batch_size
        N = self.config.n_genes
        D = self.config.dim
        V = self.config.vocab_size
        H = self.config.n_head
        
        # Embedding operations
        token_emb_flops = B * N * V  # Lookup operation (simplified)
        pos_emb_flops = B * N * D  # Addition
        
        # Attention operations per layer
        qkv_proj_flops = B * N * D * (D + 2 * D // H)  # Q: D*D, KV: D*(2*D/H)
        attention_flops = B * H * N * N * (D // H)  # Scaled dot product
        attn_out_flops = B * N * D * D  # Output projection
        
        # FFN operations per layer
        ffn_flops = B * N * D * (D * self.config.ffn_mult * 2)  # Two linear layers
        
        # Total per layer
        layer_flops = qkv_proj_flops + attention_flops + attn_out_flops + ffn_flops
        
        # Total model
        total_flops = (token_emb_flops + pos_emb_flops + 
                      self.config.n_layer * layer_flops + 
                      B * N * D * V)  # Output head
        
        # Memory requirements
        model_memory = self.config.n_params * 4  # FP32 in bytes
        activation_memory = B * N * D * self.config.n_layer * 4  # Activations
        total_memory = model_memory + activation_memory
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # FLOP breakdown
        ax = axes[0, 0]
        flop_components = {
            'Embeddings': token_emb_flops + pos_emb_flops,
            'Attention (all layers)': self.config.n_layer * (qkv_proj_flops + attention_flops + attn_out_flops),
            'FFN (all layers)': self.config.n_layer * ffn_flops,
            'Output Head': B * N * D * V
        }
        
        ax.pie(flop_components.values(), labels=flop_components.keys(), autopct='%1.1f%%', startangle=90)
        ax.set_title(f'FLOP Distribution\nTotal: {total_flops/1e9:.1f} GFLOPs', weight='bold')
        
        # Memory breakdown
        ax = axes[0, 1]
        memory_components = {
            'Model Parameters': model_memory,
            'Activations': activation_memory,
        }
        
        ax.pie(memory_components.values(), labels=memory_components.keys(), autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Memory Usage\nTotal: {total_memory/1024**3:.1f} GB', weight='bold')
        
        # Scaling analysis
        ax = axes[1, 0]
        seq_lengths = np.array([500, 1000, 2000, 4000, 8000])
        attention_scaling = seq_lengths ** 2 * D * H  # O(N^2) scaling
        ffn_scaling = seq_lengths * D * D * self.config.ffn_mult  # O(N) scaling
        
        ax.plot(seq_lengths, attention_scaling / 1e6, 'o-', label='Attention O(N²)', linewidth=2)
        ax.plot(seq_lengths, ffn_scaling / 1e6, 's-', label='FFN O(N)', linewidth=2)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('FLOPs (Millions)')
        ax.set_title('Computational Scaling', weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Batch size vs memory
        ax = axes[1, 1]
        batch_sizes = np.array([1, 4, 8, 16, 32, 64])
        memory_scaling = batch_sizes * N * D * self.config.n_layer * 4 / 1024**3  # GB
        
        ax.plot(batch_sizes, memory_scaling, 'o-', linewidth=2, markersize=8)
        ax.axhline(y=total_memory/1024**3, color='red', linestyle='--', 
                  label=f'Current config ({B} batch)')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Activation Memory (GB)')
        ax.set_title('Memory Scaling with Batch Size', weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'computational_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save computational analysis as text
        with open(self.output_dir / 'computational_analysis.txt', 'w') as f:
            f.write("Computational Requirements Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("FLOP ANALYSIS (per forward pass):\n")
            f.write(f"  Total FLOPs: {total_flops:,} ({total_flops/1e9:.1f} GFLOPs)\n")
            for comp, flops in flop_components.items():
                f.write(f"  {comp}: {flops:,} ({flops/1e9:.1f} GFLOPs)\n")
            
            f.write("\nMEMORY ANALYSIS:\n")
            f.write(f"  Model parameters: {model_memory/1024**3:.2f} GB\n")
            f.write(f"  Activation memory: {activation_memory/1024**3:.2f} GB\n")
            f.write(f"  Total memory: {total_memory/1024**3:.2f} GB\n")
            
            f.write("\nSCALING PROPERTIES:\n")
            f.write(f"  Attention complexity: O(N²) where N = {N}\n")
            f.write(f"  FFN complexity: O(N)\n")
            f.write(f"  Memory scales linearly with batch size\n")
            
            f.write("\nHARDWARE RECOMMENDATIONS:\n")
            if total_memory/1024**3 < 16:
                f.write(f"  GPU: RTX 4090 (24GB) or similar\n")
            elif total_memory/1024**3 < 40:
                f.write(f"  GPU: A100 (40GB) or H100\n")
            else:
                f.write(f"  GPU: Multi-GPU setup required\n")
    
    def create_visualization_readme(self):
        """Create a README file explaining all generated visualizations."""
        readme_content = f"""# Model Architecture Visualizations

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
**Parameters**: {self.config.n_params:,}
**Model Size**: ~{self.config.n_params * 4 / 1024**2:.1f} MB (FP32)

### Key Components:
1. **Token Embeddings**: Maps expression bins to {self.config.dim}D vectors
2. **Conditioning**: Perturbation (gene, sign, magnitude) + control set encoding
3. **Transformer Stack**: {self.config.n_layer} layers with Multi-Query Attention
4. **Diffusion Process**: {self.config.n_timesteps} timesteps with {self.config.mask_ratio} masking
5. **Output Head**: Predicts original tokens for masked positions

### Conditioning Types:
- **Target Gene**: Which gene is perturbed ({self.config.n_total_genes:,} gene vocabulary)
- **Perturbation Sign**: Direction (-1: knockdown, 0: control, +1: activation)  
- **Perturbation Magnitude**: Continuous log2 fold change value
- **Control Set**: Reference cells for comparison (encoded via set transformer)

### Training Strategy:
- **Pretraining**: {self.config.pretrain_epochs} epochs on diverse scRNA-seq data
- **Fine-tuning**: {self.config.finetune_epochs} epochs on perturbation data
- **Diffusion**: Partial masking with curriculum learning
- **Loss**: Cross-entropy on masked positions

## Usage Notes

1. **Input Format**: Tokenized gene expression ({self.config.vocab_size} bins)
2. **Output Format**: Logits over vocabulary for each gene position
3. **Inference**: Iterative denoising from fully masked input
4. **Conditioning**: Optional - model can run unconditionally

## Implementation Details

- **Attention**: Multi-Query Attention for efficiency
- **Memory**: Scales linearly with batch size, quadratically with gene count
- **Precision**: Supports FP16 training for memory efficiency
- **Hardware**: Requires ~{self.config.n_params * 4 / 1024**3:.1f} GB GPU memory for training

Generated on: {np.datetime64('today')}
Model Configuration: Conditional Diffusion Transformer v1.0
"""
        
        with open(self.output_dir / 'README.md', 'w') as f:
            f.write(readme_content)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize conditional diffusion transformer architecture'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='visualizations',
        help='Directory to save visualizations (default: visualizations)'
    )
    parser.add_argument(
        '--config-preset',
        type=str,
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Model size preset (default: medium)'
    )
    
    args = parser.parse_args()
    
    # Create config based on preset
    if args.config_preset == 'small':
        config = ConditionalModelConfig(
            dim=256, n_head=4, n_layer=4, n_genes=1000, 
            batch_size=16, n_timesteps=10
        )
    elif args.config_preset == 'large':
        config = ConditionalModelConfig(
            dim=1024, n_head=16, n_layer=12, n_genes=4000,
            batch_size=64, n_timesteps=20
        )
    else:  # medium (default)
        config = ConditionalModelConfig()
    
    # Create visualizer and generate all visualizations
    visualizer = ModelVisualizer(config, args.output_dir)
    visualizer.create_all_visualizations()
    
    print(f"\n✅ Model visualization complete!")
    print(f"📁 All files saved to: {args.output_dir}")
    print(f"📖 See {args.output_dir}/README.md for details")


if __name__ == "__main__":
    main()
