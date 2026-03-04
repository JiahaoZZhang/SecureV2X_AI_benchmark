from typing import Mapping, Callable, Optional, Dict, Iterable
from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
from abench.store.data_management import apply_perturbations

def gaussian_noise_perturbator(
    df: pd.DataFrame,
    *,
    x_col: str = "positionX",
    y_col: str = "positionY",
    sigma: float = 1.0,
    check_col: str = None,
    allowed_values: list = None) -> pd.DataFrame:
    """
    Add Gaussian noise to x and y columns, but only if the values of `check_col`
    are all contained in `allowed_values`. Otherwise return df unchanged.
    """
    # Validation des paramètres
    if check_col is not None and allowed_values is not None:
        invalid = ~df[check_col].isin(allowed_values)
        if invalid.any():
            # Retour sans perturbation
            return df

    # Application du bruit
    df = df.copy()
    df[x_col] += np.random.normal(0, sigma, size=len(df))
    df[y_col] += np.random.normal(0, sigma, size=len(df))
    return df


def piecewise_linear_perturbator(
    df: pd.DataFrame,
    *,
    x_col: str = "positionX",
    y_col: str = "positionY",
    anchor_indices: Optional[Iterable[int]] = None,
    n_segments: int = 5,
    check_col: str = None,
    allowed_values: list = None) -> pd.DataFrame:
    """
    Remplace la trajectoire (x_col, y_col) par une interpolation linéaire
    par morceaux définie sur une liste d'indices d'ancrage.
    
    - Si `anchor_indices` est fourni, on utilise exactement ces indices (bornés à [0, n-1]).
    - Sinon, on découpe la trajectoire en `n_segments` segments de longueur ~égale.
    
    La forme de sortie est identique: mêmes colonnes, même nombre de lignes.
    """
    if check_col is not None and allowed_values is not None:
        invalid = ~df[check_col].isin(allowed_values)
        if invalid.any():
            # Retour sans perturbation
            return df
        
    df = df.copy()
    n = len(df)
    if n == 0:
        return df

    # Récupération des données sous forme numpy
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    # --- Détermination des ancres ---
    if anchor_indices is not None:
        # On nettoie / borne les indices fournis
        anchors = sorted(set(i for i in anchor_indices if 0 <= i < n))
        if not anchors:
            # Fallback si tout est out-of-range
            anchors = [0, n - 1]
    else:
        # Ancres régulières: n_segments segments => n_segments+1 ancres
        if n_segments < 1:
            return df
        anchors = np.linspace(0, n - 1, n_segments + 1)
        anchors = np.round(anchors).astype(int).tolist()

    # On s'assure que la première ancre est 0 et la dernière n-1
    if anchors[0] != 0:
        anchors = [0] + anchors
    if anchors[-1] != n - 1:
        anchors = anchors + [n - 1]

    # On retire des doublons éventuels et on re-trie
    anchors = sorted(set(anchors))

    # --- Interpolation linéaire par segments ---
    new_x = np.empty_like(x, dtype=float)
    new_y = np.empty_like(y, dtype=float)

    for k in range(len(anchors) - 1):
        start = anchors[k]
        end = anchors[k + 1]

        if end <= start:
            # Cas pathologique (indices mal ordonnés), on saute
            continue

        # Valeurs aux ancres
        x0, x1 = x[start], x[end]
        y0, y1 = y[start], y[end]

        # Paramètre t allant de 0 à 1 sur le segment [start, end]
        length = end - start
        t = np.linspace(0.0, 1.0, length + 1)

        new_x[start:end + 1] = x0 + (x1 - x0) * t
        new_y[start:end + 1] = y0 + (y1 - y0) * t

    df[x_col] = new_x
    df[y_col] = new_y
    return df