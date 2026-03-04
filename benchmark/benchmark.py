import os, warnings
import numpy as np
import time

# Suppress TensorRT warnings
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

CUDA_VISIBLE_DEVICES=""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
import sys
import abench
import yaml
from pathlib import Path

# Specification of Data :
from src.data_loader import ABLoaderFromSequenceFolder,ABCVDataExperiment,TemporalFeatureScaler

def main():
    config_benchmark_path = Path(sys.argv[1]).expanduser().resolve()
    with open(config_benchmark_path) as f:
        config_benchmark = yaml.safe_load(f)
    print(config_benchmark)
    
    # Spécification of DataExperiment Plan
    from src.data_loader import get_DataExperiment
    DataExperiment = get_DataExperiment(config_benchmark)
    
    # Spécification of Models Candidates
    from src.component import build_params,get_model_constructor
    
    dict_comp = {}
    for name,config_path in config_benchmark['Components_config'].items():
        dict_comp[name]={'module':get_model_constructor(name),'parameters':build_params(config_path)}
    
    # Specification of the Component candidate list :
    exp_design=[]
    for key,model_builder in dict_comp.items():
        subexp_design=[{'name':key,'model':key}]
        exp_design.append(subexp_design)
    
    from src.component import ComponentAE
    dict_exp={'Component': ComponentAE,
              'tuning_scheme' : {},
              'model': dict_comp,
              'exp_design':exp_design}
    
    # Metric Specification 
    from src.metric import base_rmse,ABMetricGeneric
    AB_RMSE = ABMetricGeneric(metric=base_rmse,name="rmse", mask=None, dim_mask=None, list_ctx_constraint=None,reduce=True)
    list_metrics = [AB_RMSE]
    from abench.benchmark.benchmark import benchmark
    storing = 'Results'
    benchmark(storing=storing,
              ABDataExperiment=DataExperiment,
              dict_exp=dict_exp,
              list_metrics=list_metrics,verbose=True)

if __name__ == "__main__":
    main()