import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def temporal_train_test_split(df, time_col="Time", class_col="Class", train_frac=0.8):
    df = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()
    X_train = train_df.drop(columns=[class_col])
    y_train = train_df[class_col]
    X_test = test_df.drop(columns=[class_col])
    y_test = test_df[class_col]
    return X_train, y_train, X_test, y_test

def rolling_auc(y_true, y_pred, window=2000):
    scores = []
    for i in range(window, len(y_true)):
        auc = roc_auc_score(y_true[i-window:i], y_pred[i-window:i])
        scores.append(auc)
    return np.array(scores)

def train_logistic_regression(X_train, y_train, class_weight=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight=class_weight)
    model.fit(X_scaled, y_train)
    return model, scaler

def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        max_depth=5,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, scaler, X_test, y_test):
    X_input = scaler.transform(X_test) if scaler else X_test
    y_pred_prob = model.predict_proba(X_input)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    brier = brier_score_loss(y_test, y_pred_prob)
    return auc, brier, y_pred_prob
