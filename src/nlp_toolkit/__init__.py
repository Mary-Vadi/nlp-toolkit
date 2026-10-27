"""Text cleaning and dataset preparation utilities for NLP."""

from . import cleaning, dataset, eda, features, preprocessing
from .pipeline import PRESETS, CleaningPipeline

__version__ = "0.1.0"
__all__ = [
    "CleaningPipeline", "PRESETS",
    "cleaning", "preprocessing", "dataset", "eda", "features",
]
