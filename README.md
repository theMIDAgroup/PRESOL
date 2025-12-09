# PRESOL

Presol is a project supported by the Fondazione ICSC, Spoke 3 Astrophysics and Cosmos Observations. It is part of the National Recovery and Resilience Plan (Piano Nazionale di Ripresa e Resilienza, PNRR), Project ID CN_00000013 Italian Research Center on High-Performance Computing, Big Data and Quantum Computing, funded by MUR — Mission 4, Component 2, Investment 1.4: Strengthening research infrastructures and creating national R&D champions (M4C2-19) — Next Generation EU (NGEU).

The project leverages the web-based, installation-free platform HoBStudio, which hosts a Python pipeline for forecasting flare occurrence using features extracted from magnetograms.

This repository contains the same Python code and dataset available online at https://hobstudio.eu/presol

The dataset was created by downloading feature data from the HMI Active Region Patch (SHARP) FITS files provided by the Joint Science Operations Center (JSOC) and labeling them using flare event data from the GOES satellites.
A requirements.txt file is included in the repository.

# Usage

- The code is written in Python 3.10.

- All required packages and their versions are listed in requirements.txt.

- Users can run demo.py to train the model and configure the algorithm and dataset splitting. Configuration is handled through two JSON files:

- config.json — algorithm configuration

- parameters.json — dataset splitting configuration

- The demo_predict.py file is used to perform predictions.
