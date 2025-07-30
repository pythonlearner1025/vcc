# TODO
- [ ] More efficient HVG calculation over large datasets
- [ ] Technical batch encoding

# Experiment Queue
- [ ] Different mixes of masking for pretraining
- [ ] Different mixes of masking for perturbation (since zero-shot prediction begins from full masking, some full masking would help)
- [ ] Use ESM embeddings for gene 
- [ ] Auxillary loss for heteroscedastic HVG counts (Maximum Mean Discrepancy)
- [ ] Different architecture

# Installation

setup.sh 

# Data Download

- Download pretraining data scBaseCount with dataset/download.py
- Download VCC data with gdown utility from google drive link [ADD LINK]

# HVG Build

Building the HVG takes very long and requires high RAM (150GB+) for a dataset with ~300,000 cells. Algorithms for more efficient HVG calculation or moving away from it entirely should be considered.  

### Method SCRAN

Computes HVGs over only the set of intersecting genes in scRNA and VCC 

- scripts/scran_hvg.py

### Method Seurat V3

Computes HVGs over only the set of intersecting genes in scRNA and VCC 

- scritps/seurat_hvg.py