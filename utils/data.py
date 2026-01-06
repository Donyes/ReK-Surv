"""
Data loading utilities for survival analysis datasets.

Supports multiple medical datasets (METABRIC, SUPPORT, WHAS, RGBSG) and custom datasets.
"""

import numpy as np
import pandas as pd
import h5py
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


def load_h5_dataset(
    data_path: str,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load any HDF5 survival dataset with standard structure.
    
    Works for: METABRIC, SUPPORT, WHAS, RGBSG
    
    Args:
        data_path: Path to the HDF5 data file
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test: Feature matrices (standardized)
        T_train, T_test: Survival times
        E_train, E_test: Event indicators
    """
    path = pathlib.Path(data_path)
    
    with h5py.File(str(path), 'r') as f:
        X_train = f['train']['x'][()]
        E_train = f['train']['e'][()].flatten()
        T_train = f['train']['t'][()].flatten()
        
        X_test = f['test']['x'][()]
        E_test = f['test']['e'][()].flatten()
        T_test = f['test']['t'][()].flatten()
    
    # Standardize features
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Sort by survival time (descending) - required for Cox loss
    sort_train = np.argsort(-T_train)
    X_train, T_train, E_train = X_train[sort_train], T_train[sort_train], E_train[sort_train]
    
    sort_test = np.argsort(-T_test)
    X_test, T_test, E_test = X_test[sort_test], T_test[sort_test], E_test[sort_test]
    
    return X_train, X_test, T_train, T_test, E_train, E_test


def load_metabric(data_path: str = 'data/metabric.h5', random_state: int = 42):
    """Load METABRIC breast cancer dataset."""
    return load_h5_dataset(data_path, random_state)


def load_support(data_path: str = 'data/support.h5', random_state: int = 42):
    """Load SUPPORT ICU mortality dataset."""
    return load_h5_dataset(data_path, random_state)


def load_whas(data_path: str = 'data/whas.h5', random_state: int = 42):
    """Load WHAS heart attack dataset."""
    return load_h5_dataset(data_path, random_state)


def load_rgbsg(data_path: str = 'data/rgbsg.h5', random_state: int = 42):
    """Load RGBSG breast cancer dataset."""
    return load_h5_dataset(data_path, random_state)


def load_hlb(
    data_path: str = 'data/hlb.xlsx',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load HLB (Huanglongbing/Citrus Greening) survival dataset.
    
    Args:
        data_path: Path to the Excel data file
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test: Feature matrices (standardized)
        T_train, T_test: Survival times
        E_train, E_test: Event indicators (1=infected, 0=censored)
    """
    df = pd.read_excel(data_path)
    
    # Extract features and targets
    X = df.drop(columns=['T', 'E', '编号']).values
    T = df['T'].values
    E = df['E'].values
    
    return load_custom_dataset(X, T, E, test_size, random_state)


def load_custom_dataset(
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess a custom survival dataset.
    
    Args:
        X: Feature matrix, shape (n_samples, n_features)
        T: Survival times, shape (n_samples,)
        E: Event indicators, shape (n_samples,)
        test_size: Fraction of data for testing
        random_state: Random seed
    
    Returns:
        X_train, X_test, T_train, T_test, E_train, E_test
    """
    # Stratified split based on event status and time quantiles
    stratify_by = []
    for e, t in zip(E, T):
        if e == 1:
            time_bin = np.digitize(t, bins=np.percentile(T[E == 1], [33, 66]))
            stratify_by.append(f"event_{time_bin}")
        else:
            stratify_by.append("non_event")
    
    X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
        X, T, E,
        test_size=test_size,
        stratify=stratify_by,
        random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Sort by survival time (descending)
    sort_train = np.argsort(-T_train)
    X_train, T_train, E_train = X_train[sort_train], T_train[sort_train], E_train[sort_train]
    
    sort_test = np.argsort(-T_test)
    X_test, T_test, E_test = X_test[sort_test], T_test[sort_test], E_test[sort_test]
    
    return X_train, X_test, T_train, T_test, E_train, E_test


def prepare_tensors(
    X_train: np.ndarray,
    X_test: np.ndarray,
    T_train: np.ndarray,
    T_test: np.ndarray,
    E_train: np.ndarray,
    E_test: np.ndarray
) -> dict:
    """
    Convert numpy arrays to PyTorch tensors.
    
    Returns:
        Dictionary containing all tensors
    """
    import torch
    
    return {
        'X_train': torch.tensor(X_train, dtype=torch.float32),
        'X_test': torch.tensor(X_test, dtype=torch.float32),
        'T_train': torch.tensor(T_train, dtype=torch.float32),
        'T_test': torch.tensor(T_test, dtype=torch.float32),
        'E_train': torch.tensor(E_train, dtype=torch.float32),
        'E_test': torch.tensor(E_test, dtype=torch.float32),
    }
