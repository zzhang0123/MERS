# MERS: Moment Expanded Radio Sky

A framework for modeling radio foregrounds using power-law ensemble moment expansion formalism for 21cm cosmology applications.

## Overview

This repository implements and analyzes radio foreground modeling techniques using a power-law ensemble moment expansion formalism. 

## Project Structure

The analysis is organized into three main components:

### 1. Diffuse Foreground Model Comparison
**Notebook:** `ME_diffuse_sky_map.ipynb`

Comparative analysis of CNN-PL vs GSM models using full-sky maps:
- **Loss map comparisons**: Evaluate fitting performance across different maximum moment orders
- **Moment map comparisons**: Compare spatial distributions of moment coefficients between sky models

### 2. Beam Chromaticity and Point Source Impact Analysis  
**Notebook:** `ME_TOD_beam_ptsrc.ipynb`

Time-ordered data (TOD) analysis examining various observational scenarios:
- Ideal case: no point sources, no beam effects
- Gaussian achromatic beam
- Sinc² achromatic beam  
- Sinc² chromatic beam

### 3. Global 21cm Foreground Modeling
**Notebook:** `ME_global21_fg.ipynb`

Advanced foreground mitigation techniques for global 21cm experiments using Sinc² chromatic beam:
- **BCF method**: Beam Convolution Function approach (implemented, not good)
- **SVD method**: Singular Value Decomposition beam modeling (implemented, not good)  
- **Map-making approach**: Monopole reconstruction using limTOD (TBD)
- ...



