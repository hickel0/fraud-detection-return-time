# Fraud Detection - Return Time Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)](.)

Machine learning pipeline for fraud detection focusing on return time prediction and imbalanced classification techniques.

## Project Overview

**Course**: CSC1107 - Application Domains of Machine Learning
**Focus**: Fraud Detection & Imbalanced Learning
**Status**: Completed

This project implements fraud detection algorithms with emphasis on:
- Return time prediction for fraudulent transactions
- Pattern analysis vs. classification approaches
- Handling highly imbalanced datasets

## Repository Structure

```
App Domains/
├── Project/
│   ├── Q1_RT_Prediction.ipynb              # Return time prediction
│   ├── Q2_Imbalance_Handling_Methods.ipynb # Imbalance techniques comparison
│   ├── Q3_patterns_vs_classification.ipynb # Pattern vs classification analysis
│   ├── ML_Analysis_Notebook.ipynb          # Main analysis notebook
│   ├── Data Import from Kagglehub.ipynb    # Dataset loading
│   └── Basic_Analysis.ipynb                # Initial exploration
├── Lee_Hickey_A21407166/
│   ├── Code/
│   │   └── ML_Analysis_Notebook.ipynb      # Final submission notebook
│   ├── README.md                           # Setup instructions
│   ├── download_data.py                    # Data download script
│   ├── requirements.txt                    # Dependencies
│   └── src/
│       └── pipeline_utils.py               # Utility functions
├── Lectures/                               # Course materials
├── Papers/                                 # Reference papers
└── Hickey_A21407166_CSC1107.pdf           # Final report
```

## Key Research Questions

### Q1: Return Time Prediction
Can we predict when a fraudulent transaction will occur based on historical patterns?

### Q2: Imbalance Handling Methods
Comprehensive comparison of techniques:
- SMOTE (Synthetic Minority Over-sampling)
- Random Undersampling
- Class Weight Adjustment
- Ensemble Methods (BalancedRandomForest)
- Threshold Optimization

### Q3: Patterns vs Classification
Comparing rule-based pattern detection against ML classification approaches for fraud detection.

## Technologies Used

- **Python 3.10+**
- **scikit-learn** - ML algorithms and evaluation
- **XGBoost** - Gradient boosting classifier
- **imbalanced-learn** - SMOTE and resampling techniques
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **KaggleHub** - Dataset access

## Setup & Installation

```bash
# Navigate to project
cd "App Domains/Lee_Hickey_A21407166"

# Install dependencies
pip install -r requirements.txt

# Download dataset
python download_data.py

# Run analysis
jupyter notebook Code/ML_Analysis_Notebook.ipynb
```

## Results

- Successfully compared multiple imbalance handling techniques
- Demonstrated trade-offs between precision and recall
- Analyzed pattern-based vs. ML classification approaches
- Reproducible pipeline with comprehensive documentation

## Key Learnings

- Importance of proper evaluation metrics for imbalanced data (F1, PR-AUC vs Accuracy)
- SMOTE effectiveness depends on dataset characteristics
- Ensemble methods often outperform single classifiers on imbalanced data
- Pattern analysis complements but doesn't replace ML classification

## Author

**Lee Hickey** - Dublin City University
Student ID: A21407166

---

*Academic Project - CSC1107 Application Domains of Machine Learning*
