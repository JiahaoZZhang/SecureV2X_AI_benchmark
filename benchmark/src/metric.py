import numpy as np
from abench.metric.metric import ABMetricGeneric
from uqmodels.evaluation.metrics import base_rmse

def recon_rmse(y, pred, **kwarg):
    """Root mean square for nD array

    Args:
        y (np.array): Targets/observation
        pred (np.array): Prediction/Reconstruction

    Returns:
        val: rmse values
    """
    val = np.sqrt(np.power(y-pred,2).mean(axis=1))
    return val


def recon_mae(y, pred, **kwarg):
    """Mean absolute error for nD array

    Args:
        y (np.array): Targets/observation
        pred (np.array): Prediction/Reconstruction

    Returns:
        val: mea values
    """
    val = np.abs(y-pred).mean(axis=(0,1))
    return val
