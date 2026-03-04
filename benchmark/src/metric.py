import numpy as np
from abench.metric.metric import ABMetricGeneric
from uqmodels.evaluation.metrics import base_rmse

def recon_score(y, pred, **kwarg):
    """Root mean square for nD array

    Args:
        y (np.array): Targets/observation
        pred (np.array): Prediction/Reconstruction

    Returns:
        val: rmse values
    """
    val = np.sqrt(np.power(y-pred,2).mean(axis=(1,2)))
    return val