import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, Any
import abench.store.api as api
from abench.visu.visu import plot_grid_mean_std
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix, roc_curve


def plot_trajet(
    ax,
    y: np.ndarray,
    output: np.ndarray = None,
    context: Optional[np.ndarray] = None,
    metadata: Optional[Any] = None,
    lim: Tuple[float, float] = None,
    colors: Tuple[str, str] = ("green", "red"),
    point_size: int = 10,
    line_width: float = 1.2):
    """
    Visualize a single (y, output) trajectory pair on one Matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target subplot to draw on.
    y : np.ndarray
        Ground-truth trajectory, shape (T, 2).
    output : np.ndarray
        Predicted or reconstructed trajectory, shape (T, 2).
    context : np.ndarray | None
        Optional conditioning input (unused here but kept for API consistency).
    metadata : Any | None
        Optional sample metadata, e.g., ID or label.
    lim : tuple(float, float)
        Plot limits for both x and y axes.
    colors : tuple(str, str)
        Colors for (ground truth, prediction).
    point_size : int
        Size of scatter points.
    line_width : float
        Line width for plotted trajectories.
    """
    y = np.asarray(y)
    c_real, c_pred = colors
    # Ground truth trajectory
    ax.scatter(y[:, 0], y[:, 1], s=point_size, color=c_real, alpha=0.7)
    ax.plot(y[:, 0], y[:, 1], color=c_real, linewidth=line_width, label="Ground truth")

    # Predicted trajectory
    if(output is not None):
        output = np.asarray(output)
        ax.scatter(output[:, 0], output[:, 1], s=point_size, color=c_pred, alpha=0.7)
        ax.plot(output[:, 0], output[:, 1], color=c_pred, linewidth=line_width, linestyle="--", label="Predicted")
    if(lim is not None):
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if metadata is not None:
        ax.set_title(f"Sample {metadata}")

    ax.legend(fontsize=8, loc="upper right")
    
    
    

def plot_tra(
    ax,
    y: np.ndarray,
    output: np.ndarray = None,
    context: Optional[np.ndarray] = None,
    metadata: Optional[Any] = None,
    lim: Tuple[float, float] = None,
    colors: Tuple[str, str] = ("green", "red"),
    point_size: int = 10,
    line_width: float = 1.2):
    
    
    
    y = np.asarray(y)
    c_real, c_pred = colors
    # Ground truth trajectory
    ax.scatter(y[:, 0], y[:, 1], s=point_size, color=c_real, alpha=0.7)
    ax.plot(y[:, 0], y[:, 1], color=c_real, linewidth=line_width, label="Ground truth")

    # Predicted trajectory
    if(output is not None):
        output = np.asarray(output)
        ax.scatter(output[:, 0], output[:, 1], s=point_size, color=c_pred, alpha=0.7)
        ax.plot(output[:, 0], output[:, 1], color=c_pred, linewidth=line_width, linestyle="--", label="Predicted")
    if(lim is not None):
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if metadata is not None:
        ax.set_title(f"Sample {metadata}")

    ax.legend(fontsize=8, loc="upper right")





def computeGridResults(dict_perf,list_component_name,list_metrics,dict_experiments,ctx_1,ctx_2,n_REPET):
	dict_mat = {}
	for metric in list_metrics:
		dict_mat[metric]=np.zeros((len(dict_experiments),len(list_component_name),len(ctx_1),len(ctx_2),2))
		for na,(name_exp,experiment_plan) in enumerate(dict_experiments.items()):
			print(experiment_plan)
			metric_dict_to_plot = api.extract_benchmark_tables(dict_perf,experiment_plan=experiment_plan,list_components_name=list_component_name,
									metrics=list_metrics,agg_name='no-agg')
			for nb,model_name in enumerate(list_component_name):
				dict_mat[metric][na,nb,:,:,0] = np.array(metric_dict_to_plot[metric][model_name]).reshape(n_REPET,len(ctx_1),len(ctx_2)).mean(axis=0)
				dict_mat[metric][na,nb,:,:,1] = np.array(metric_dict_to_plot[metric][model_name]).reshape(n_REPET,len(ctx_1),len(ctx_2)).std(axis=0)
	return(dict_mat)

def plot_CondGridResults(dict_perf,list_component_name,list_metrics,dict_experiments,ctx_1,ctx_2,n_REPET,save_name='figure'):
	list_exp_name = list(dict_experiments.keys())
	dict_mat = computeGridResults(dict_perf,list_component_name,list_metrics,dict_experiments,ctx_1,ctx_2,n_REPET)
	for metric in list_metrics:
		fig,axes = plot_grid_mean_std(means= dict_mat[metric][0:1,:,:,:,0],
								stds=None,
								cmap='RdYlGn_r',
								meta_col_headers=list_component_name,
								meta_row_headers=list_exp_name[0:1],
								col_labels=ctx_2,
								row_labels=ctx_1,
								tick_fontsize=20,
								axis_label_fontsize=20,
								annotate_fontsize=40,
								value_fmt="{:.4f}",
                                colorbar_label="RMSE",
								figsize_per_cell=(5,2))
		# fig.savefig(save_name+metric+'_1')
		fig,axes = plot_grid_mean_std(means= dict_mat[metric][1:,:,:,:,0]-dict_mat[metric][0:1,:,:,:,0],
								stds= None,
								cmap='seismic',
								meta_col_headers=list_component_name,
								meta_row_headers=list_exp_name[1:],
								row_labels=ctx_1,
								col_labels=ctx_2,
								tick_fontsize=20,
								axis_label_fontsize=20,
								annotate_fontsize=40,
								figsize_per_cell=(5,2),
								value_fmt="{:.4f}",
                                colorbar_label="Delta RMSE",
								vmin_mirror=True)
		# fig.savefig(save_name+metric+'_2')





def thresholdTunning(df, iterations, *, threshold_val=None):
    """threshold tunning

    Args:
        df (DataFrame): prediction result and original result
        iterations (int): iteration number
        threshold_val (float, optional): fixed threshold value. Defaults to None.

    Returns:
        object: All metrics
    """
    thresh_df = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1-score': []
    }
    
    if threshold_val is not None:
            preds = df['error'] > threshold_val
            cr = classification_report(df['y_true'], preds, output_dict=True)
            acc = cr['accuracy']
            prec = cr['macro avg']['precision']
            rc = cr['macro avg']['recall']
            f1 = cr['macro avg']['f1-score']
            thresh_df['threshold'].append(threshold_val)
            thresh_df['accuracy'].append(acc)
            thresh_df['precision'].append(prec)
            thresh_df['recall'].append(rc)
            thresh_df['f1-score'].append(f1)

            print(f"Threshold: {threshold_val:.4f}\tAccuracy: {acc:.3f}\t\tPrecision: {prec:.3f}\tRecall Score: {rc:.3f}\tf1-score: {f1:.3f}")
    else:
        for i in range(iterations):
            thresh_value = df['error'].quantile(i/iterations)
            preds = df['error'] > thresh_value
            cr = classification_report(df['y_true'], preds, output_dict=True)
        
            acc = cr['accuracy']
            prec = cr['macro avg']['precision']
            rc = cr['macro avg']['recall']
            f1 = cr['macro avg']['f1-score']
            
            thresh_df['threshold'].append(thresh_value)
            thresh_df['accuracy'].append(acc)
            thresh_df['precision'].append(prec)
            thresh_df['recall'].append(rc)
            thresh_df['f1-score'].append(f1)
            
            print(f"Threshold: {thresh_value:.4f}\tAccuracy: {acc:.3f}\t\tPrecision: {prec:.3f}\tRecall Score: {rc:.3f}\tf1-score: {f1:.3f}")
        
    return pd.DataFrame(thresh_df)