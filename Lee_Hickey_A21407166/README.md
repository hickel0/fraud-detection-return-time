# Real-Time Fraud Detection – Reproducible Pipeline

This project implements a fully reproducible temporal fraud-detection pipeline for CSC1107.

## 1. Setup Environment
Install dependencies:
    pip install -r requirements.txt

## 2. Download Dataset Automatically
Ensure Kaggle API credentials exist, then run:
    python download_data.py

## 3. Running the Notebook
Open Jupyter:
    jupyter notebook
Run Fraud_Detection_Pipeline.ipynb

## 4. Structure
project/
 ├── data/
 ├── src/
 │    └── pipeline_utils.py
 ├── notebooks/
 ├── requirements.txt
 ├── download_data.py
 └── README.md
