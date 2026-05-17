"""
Preprocessing utilities for the TPE4 datasets.

Provides standardization (zero mean, unit variance per column) and a
convenience loader for the Europe dataset, both of which are shared by
Kohonen and Oja.
"""

import numpy as np
import pandas as pd


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std


def load_europe(path="data/europe.csv"):
    df = pd.read_csv(path)
    countries = df["Country"].values
    variables = df.columns[1:].tolist()
    X_raw = df.drop(columns=["Country"]).values
    X, _, _ = standardize(X_raw)
    return countries, X, variables