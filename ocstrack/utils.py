"""Utility functions"""

import logging
from typing import Union

import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
_logger = logging.getLogger()


def convert_longitude(lon: Union[float, np.ndarray],
                      mode: int = 1) -> np.ndarray:
    """
    Convert longitudes between common geographic conventions.

    Args:
        lon: array-like of longitudes
        mode: conversion mode:
            - 1: Convert from [-180, 180] (Greenwich at 0°) → [0, 360] (Greenwich at 0°)
            - 2: Convert from [0, 360] (Greenwich at 0°) → [-180, 180] (Greenwich at 0°)
            - 3: Convert from [-180, 180] (Greenwich at 0°) → [0, 360] (Greenwich at 180°)
            - 4: Convert from [0, 360] (Greenwich at 0°) → [0, 360] (Greenwich at 180°)
            - 5: Convert from [0, 360] (Greenwich at 180°) → [0, 360] (Greenwich at 0°)

    Returns:
        np.ndarray of converted longitudes
    """
    _logger.debug("Converting longitude with mode %d", mode)
    lon = np.asarray(lon)
    if mode == 1:
        return lon % 360
    if mode == 2:
        return np.where(lon > 180, lon - 360, lon)
    if mode == 3:
        return lon + 180
    if mode == 4:
        return (lon + 180) % 360
    if mode == 5:
        return (lon - 180) % 360

    raise ValueError(f"Invalid mode {mode}. Supported modes: 1, 2, 3, 4, 5")
