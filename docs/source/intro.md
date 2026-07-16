# Introduction

<p align="center">
  <picture>
    <img alt="ATLAS: A workflow for materials NNP generation" src="../_static/logo_light.png" width="600">
  </picture>

ATLAS (Automated Training with Latent-space Aware Sampling) is a unified Python framework for building robust machine learning interatomic potentials (MLIPs). It combines a diversity-aware database generator with a manifold-aware active learning workflow to produce compact, high-quality training datasets.

Key capabilities include:

- **Structure generation** for bulk, surface, cluster, and isolated atom configurations across single-, binary-, and ternary-phase diagrams, with perturbations, vacancies, deformations, and adsorbates.
- **Active learning** with a modular backend architecture for training MLIPs. It currently supports MACE, Allegro, and NequIP, and any model can be added through the backend plugin interface. The workflow runs molecular dynamics simulations, detects extrapolating structures via descriptor-based or latent-space methods, and submits them for DFT labelling.
- **Data reduction mode** for iteratively selecting structures from large pre-existing databases.
- **Safeguard checks** to prevent premature convergence of active learning loops.
- **Diversity metrics** including Vendi Score and Circles Metric.
- **Benchmarking** tools for both MLIP and DFT evaluation.
- **Interactive monitoring** via a Flask dashboard and a PySide6 desktop GUI (ATLAS Hub).
- **Comprehensive reporting** of model performance and resource usage.

The entire workflow is orchestrated through AiiDA for reproducible and provenance-tracked execution.
