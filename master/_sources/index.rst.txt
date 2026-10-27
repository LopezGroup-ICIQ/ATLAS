.. ATLAS documentation master file, created by
   sphinx-quickstart on Tue Oct 15 17:21:48 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Intro to ATLAS
==============

.. image:: ../media/logo_dark.png
   :width: 400 px
   :alt: ATLAS: A workflow for materials MLIP generation"
   :align: center

|

ATLAS (Automated Training with Latent-space Aware Sampling) is a unified Python framework for building robust machine learning interatomic potentials (MLIPs). It combines a diversity-aware database generator with a manifold-aware active learning workflow to produce compact, high-quality training datasets. ATLAS supports structure generation for bulk, surface, cluster, and isolated atom configurations across single-, binary-, and ternary-phase diagrams, with perturbations, vacancies, deformations, and adsorbates. The active learning engine iteratively trains MACE models, runs molecular dynamics simulations, detects extrapolating structures via descriptor-based or latent-space methods (autoencoder + concave hull), and submits them for DFT labelling, all orchestrated through AiiDA. Additional capabilities include data reduction mode, safeguard checks to prevent premature convergence, test database evaluation, diversity metrics (Vendi Score, Circles Metric), an interactive monitoring dashboard, a desktop GUI (ATLAS Hub), MLIP benchmarking, and comprehensive reporting of model performance and resource usage. Validated on metals, alloys, and metal oxides, ATLAS produces datasets that exceed foundation model training chemical spaces by orders of magnitude while using up to x300 fewer structures.


.. toctree::
   :maxdepth: 2
   :caption: Usage

   source/intro.md
   source/install.md
   source/tools.md
   source/input.md
   source/customization.md

.. toctree::
   :maxdepth: 1
   :caption: Examples

   Database Generation <source/results_db_gen.md>

.. toctree::
   :maxdepth: 1
   :caption: Package information

   Package Information <source/modules.rst>
