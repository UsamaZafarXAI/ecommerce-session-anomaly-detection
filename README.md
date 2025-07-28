# Explainable Anomaly Detection in Chatbot-Driven E-Commerce Sessions

This repository contains the full pipeline described in our journal manuscript:

**Manuscript Title:**  
*Detecting Behavioral Anomalies in Chatbot-Driven E-Commerce Sessions: An Explainable Machine Learning Approach*

---

## Overview

This project identifies and explains anomalies in e-commerce sessions augmented with simulated chatbot interactions. It uses an ensemble of unsupervised models for pseudo-labeling (Isolation Forest + Autoencoder) and a supervised LightGBM classifier, supported by SHAP and LIME for interpretability.

---

## Dataset

RetailRocket Recommender System Dataset (2.8M events)  
[Kaggle Source](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

---

## Pipeline Components

1. Session segmentation (30-minute idle timeout)
2. Feature engineering (interaction counts, conversion ratios, etc.)
3. Pseudo-labeling via Isolation Forest & Autoencoder
4. Supervised classification with LightGBM
5. Explainability via SHAP and LIME
6. Statistical validation using Cliff’s Δ

---

## How to Run

### Requirements
```bash
pip install -r requirements.txt
```

### Execution
```bash
python run_pipeline.py
```

Artifacts (plots, CSVs, SHAP/LIME visuals) will be saved in `/retailrocket_outputs/`.

---

## File Structure

- `run_pipeline.py` – Entry point to execute the full framework
- `requirements.txt` – Dependencies
- `src/` – Modular Python scripts (preprocessing, modeling, XAI, etc.)
- `outputs/` – Generated artifacts (excluded from repo)
- `notebooks/` – (optional) Jupyter demos

---

## License

MIT License — Free for academic and non-commercial use.
