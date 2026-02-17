# Radius-Dependent Plasmonic Transmission in Gold Nanoparticle-Embedded Hydrogels

## Overview

This project investigates how gold nanoparticle radius influences
optical transmission in a hydrogel (contact lens-like) system
across the visible spectrum (400–800 nm).

The study combines:

- Experimental data analysis
- Visualization of radius-dependent attenuation
- Random Forest regression modeling
- Feature importance analysis

to quantify structure–optical property relationships.

## Dataset

The dataset contains 810 measurements with columns:

- `radius_nm`
- `wavelength_nm`
- `transmission`

Each row corresponds to one experimental transmission measurement.

## Key Findings

- Transmission decreases strongly with increasing nanoparticle radius.
- Radius contributes ~75% of the model importance.
- Wavelength contributes ~25%.
- The behavior is consistent with plasmonic extinction scaling.

## Methods

1. Exploratory visualization (Transmission vs Wavelength).
2. Radius-dependent transmission plots.
3. Random Forest regression model.
4. Feature importance analysis.

## Model Performance

R² ≈ 0.999  
MAE ≈ 0.005  

The high R² indicates strong structural predictability of optical transmission.

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
python main.py
