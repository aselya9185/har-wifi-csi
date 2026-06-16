# WiFi Sensing Through Digital Receive Beamforming and CSI

This repository contains the source code, experiments, visualizations, and thesis materials developed for WiFi-based Human Activity Recognition (HAR) using Channel State Information (CSI) and digital receive beamforming.

The project was developed as part of the Master's thesis:

**Assel Ussenova**
MSc in ICT and Internet Engineering
Università degli Studi di Roma Tor Vergata (2024/2025)

---

## Overview

WiFi Channel State Information (CSI) can be used as a sensing modality to recognize human activities without requiring wearable devices or cameras.

This project implements a complete CSI processing pipeline, from raw CSI preprocessing and reconstruction to digital receive beamforming, feature extraction, and machine learning-based activity classification.

The monitored activities include:

* Empty room
* Sitting
* Standing
* Walking

---

## Features

This repository implements a complete WiFi sensing pipeline for Human Activity Recognition (HAR) based on Channel State Information (CSI) and digital receive beamforming.

Key capabilities include:

* CSI preprocessing and reconstruction
* Pilot subcarrier interpolation and phase calibration
* Digital receive beamforming with antenna-pair selection
* Virtual beam scanning through phase-weight optimization
* CSI distance computation for motion characterization
* Optimal beam angle (θ) estimation and temporal analysis
* Beamforming-based feature extraction
* Random Forest-based Human Activity Recognition (HAR)
* Cross-environment evaluation across multiple indoor scenarios
* Visualization and analysis of CSI amplitudes, phases, beamforming outputs, and classification results
* Reproducible end-to-end processing pipeline from CSI measurements to activity classification

---

## Repository Structure

```text
docs/                       # Thesis documentation
plots/                      # Generated figures and visualizations
results/                    # Experimental results and confusion matrices

interpolation.py            # Pilot subcarrier interpolation
PLL_bias.py                 # Phase bias correction
LRT.py                      # Phase trend removal using Linear Regression Transformation
phase_raw.py                # Raw CSI phase visualization
phase_mean.py               # Mean CSI phase visualization and comparison
reconstruct_csi.py          # CSI reconstruction from processed phase and amplitude

beamforming.py              # Digital receive beamforming
beamforming_segments.py     # CSI distance and optimal beam-angle estimation using segmented beamforming
csi_distance.py             # CSI distance computation and analysis

feature_extraction.py       # Extraction of beamforming-based features for HAR
random_forest.py            # Activity classification using Random Forest
analyze_walking_period.py   # Walking activity analysis and visualization

plot_csi.py                # CSI visualization utilities
utils.py                   # Utility functions

requirements.txt           # Project dependencies
```

---

## Processing Pipeline

```text
Raw CSI
    ↓
Interpolation
    ↓
Phase Processing
    ↓
CSI Reconstruction
    ↓
Digital Receive Beamforming
    ↓
CSI Distance & Beam Angle Analysis
    ↓
Feature Extraction
    ↓
Random Forest Classification
```

---

## Dataset

The complete dataset, intermediate processing outputs, extracted features, plots, and experimental results are available on Hugging Face:

**Dataset:**
https://huggingface.co/datasets/aselya9185/wifi-csi-human-activities

The dataset includes:

* Raw CSI measurements
* Interpolated CSI
* Phase-processed CSI
* Reconstructed CSI
* CSI distance outputs
* Beamforming angle outputs
* Feature datasets
* Experimental results
* Generated plots

---

## Experimental Setup

CSI measurements were collected using the Nexmon CSI extraction framework on an Asus RT-AC86U router operating in IEEE 802.11ac VHT 80 MHz mode.

Experiments were conducted in two indoor office environments. Human activities including empty room, sitting, standing, and walking were recorded and analyzed using beamforming-derived CSI features.

The system was evaluated on cross-environment activity recognition tasks to assess robustness to changes in the propagation environment.

---

## Related Publication

The CSI acquisition framework and experimental campaign used in this work are based on:

**M. De Sanctis, R. Fallani, T. Rossi, E. Cianca, M. Ruggieri, and V. Poulkov**,
*"WiFi Sensing Through Digital Receive Beamforming and CSI,"*
2025 IEEE 36th International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC), 2025.
DOI: 10.1109/PIMRC62392.2025.11274992

This repository focuses on the processing, beamforming, feature extraction, analysis, and classification pipeline developed as part of the accompanying Master's thesis.

---

## Thesis

The thesis PDF is available in the `docs/` directory.

**Assel Ussenova**
*WiFi Sensing Through Digital Receive Beamforming and CSI*
Master's Thesis
Università degli Studi di Roma Tor Vergata
Academic Year 2024/2025

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/aselya9185/har-wifi-csi.git
cd har-wifi-csi
pip install -r requirements.txt
```

---

## Research Areas

* WiFi Sensing
* Human Activity Recognition (HAR)
* Channel State Information (CSI)
* RF Sensing
* Digital Receive Beamforming
* Wireless Signal Processing
* Machine Learning

---

## Citation

If you use this repository or dataset in your research, please cite:

```bibtex
@inproceedings{desanctis2025wifi,
  title={WiFi Sensing Through Digital Receive Beamforming and CSI},
  author={De Sanctis, Mauro and Fallani, Rebecca and Rossi, Tommaso and Cianca, Ernestina and Ruggieri, Marina and Poulkov, Vladimir},
  booktitle={2025 IEEE 36th International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)},
  year={2025},
  doi={10.1109/PIMRC62392.2025.11274992}
}
```

---

## License

This project is provided for research and educational purposes.
