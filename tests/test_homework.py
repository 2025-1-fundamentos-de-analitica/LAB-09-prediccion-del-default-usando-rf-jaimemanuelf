# flake8: noqa: E501
"""Autograding script."""

import gzip
import json
import os
import pickle

import pandas as pd  # type: ignore

# ------------------------------------------------------------------------------
MODEL_FILENAME = "files/models/model.pkl.gz"
MODEL_COMPONENTS = [
    "OneHotEncoder",
    "RandomForestClassifier",
]
SCORES = [
    0.785,
    0.673,
]
METRICS = [
    {
        "type": "metrics",
        "dataset": "train",
        "precision": 0.944,
        "balanced_accuracy": 0.785,
        "recall": 0.580,
        "f1_score": 0.719,
    },
    {
        "type": "metrics",
        "dataset": "test",
        "precision": 0.650,
        "balanced_accuracy": 0.673,
        "recall": 0.401,
        "f1_score": 0.498,
    },
    {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": 16060, "predicted_1": None},
        "true_1": {"predicted_0": None, "predicted_1": 2740},
    },
    {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": 6670, "predicted_1": None},
        "true_1": {"predicted_0": None, "predicted_1": 760},
    },
]


# ------------------------------------------------------------------------------
#
# Internal tests
#
def _load_model():
    """Generic test to load a model"""
    assert os.path.exists(MODEL_FILENAME)
    with gzip.open(MODEL_FILENAME, "rb") as file:
        model = pickle.load(file)
    assert model is not None
    return model





def test_homework():
    """Tests"""

    model = _load_model()
   
