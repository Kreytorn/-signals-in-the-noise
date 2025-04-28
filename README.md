# Signals in the Noise: Detecting Particles in CERN Collisions with Machine Learning

## 📚 Project Overview

This project analyzes real collision data from CERN, focusing on **dielectron events**.  
The goal is to **separate real particle signals** (like Z boson decays) from **background noise** using **machine learning**.

We predict whether a collision event is likely to be a true particle signal based on energy, momentum, and angular features.

---

## 📂 Project Structure

- **dielectron.csv** — Raw dataset containing electron collision properties.
- **signals-in-the-noise.ipynb** — Jupyter Notebook with full pipeline: loading, preprocessing, visualization, modeling.
- **lgbm_model.pkl** — Trained LightGBM model saved for future predictions.

---

## 🛠️ What We Did

- Cleaned and prepared the dataset.
- Filled missing invariant mass (`M`) values using **Nearest Neighbor** search.
- Created new physics-based features:
  - Total energy (`total_energy`)
  - Difference in transverse momentum (`delta_pt`)
  - Combined momenta (`sum_px`, `sum_py`, `sum_pz`)
  - Angular difference (`delta_eta`)
- Scaled all features for better machine learning performance.
- Labeled events:
  - **Signal** → if mass is between 85 GeV and 95 GeV.
  - **Background** → otherwise.
- Trained a **LightGBM Classifier** to predict signal vs background.

---

## 📈 Results

- **Test Accuracy:** 98.18%
- **Signal Detection (F1-Score):** 0.85
- **Most Important Features:** `total_energy`, `pt1`, `pt2`, `delta_eta`

The feature importance plot showed that physics-based features like **total_energy** were critical for model decisions, while simpler properties like **Q1** and **Q2** (charges) were not important.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Kreytorn/-signals-in-the-noise.git
