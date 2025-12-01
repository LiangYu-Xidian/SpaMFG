# SpaMFG: A Spatially-Aware Multi-Omics Feature Grouping & Integration Framework
This repository provides an end-to-end framework for integrating RNA and ATAC multi-omics data with spatial information. The method performs feature preprocessing, spatial-enhanced similarity computation, spectral clustering, cross-modality feature matching, MOFA-based integration, and downstream clustering/visualization. The implementation is designed for spatial or single-cell multi-omics datasets such as Mouse Thymus and provides a robust pipeline for identifying biologically consistent feature modules across modalities.
## Overview of the Method
### 1. High-Variable Feature Selection and Preprocessing
### 2. Spatially-Enhanced Similarity Matrix
A key idea of this framework is that gene grouping should consider both expression similarity and spatial proximity.

compute_similarity_matrix_with_spatial() computes:

1. Cosine similarity of gene expression
2. Spatial kernel similarity from coordinates
3. Weighted combination of the two
