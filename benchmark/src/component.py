import abench
from abench.component.component import GenericComponent
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd
import yaml
import uqmodels.modelization.TF_estimator.vae.vae as vae
import uqmodels.modelization.TF_estimator.vae.timevae as timevae
from uqmodels.modelization.TF_estimator.base.train_config import make_optimizer_default,make_callbacks
from uqmodels.modelization.TF_estimator.base.loss import build_MSE_loss
from copy import deepcopy

class ComponentAE(GenericComponent):
    def __init__(self,model={'initializer':None,'parameters':None}):
        super().__init__(model=model)

    def fit(self,X,y,**kwargs):
        return(super().fit(X,y,**kwargs))
        
    def predict(self,X,y=None,**kwargs):
        return(super().predict(X,y,**kwargs))
        
    def save(self,storing,keys):
        full_path = Path(storing, *keys)
        self.model.save(full_path)

    @classmethod
    def load(cls,storing,keys):
        dir_model = Path(storing, *keys)
        with open(Path(dir_model,'config.yaml')) as f:
            config_comp = yaml.safe_load(f)
        print(config_comp)
        class_name = config_comp['class_name']
        model_params = config_comp['init_kwargs']
        training_params = deepcopy(model_params['training_params'])
        model_params['training_params'] = None
        model_config = {'training_params':training_params,'model_params':model_params}
        self = cls(model={'initializer':get_model_constructor(class_name),'parameters':aux_build_params(model_config)})
        self.model = get_model_constructor(class_name).load(dir_model)
        return(self)
    
    def get_params(self):
        params = self.__dict__
        return params

def aux_build_params(model_config):
        model_params = model_config['model_params']
        training_params = model_config['training_params']
        opt = make_optimizer_default(training_params['optimizer'])
        mse = build_MSE_loss(split=1,metric=False)
        mse_metric = build_MSE_loss(split=1,metric=True)
        default_compile_kwargs = dict(optimizer=opt,loss=mse,metrics=[mse_metric])
        callbacks, configs = make_callbacks()
        if('kl_weight' in training_params):
            callbacks = add_vae_callbacks(callbacks,kl_weight)
        default_fit_kwargs = dict(epochs=training_params['epochs'],batch_size=training_params['batch_size'], verbose=1,callbacks=callbacks,shuffle=True)
        model_params['training_params']= training_params    
        model_params['compile_kwargs']= default_compile_kwargs
        model_params['fit_kwargs']= default_fit_kwargs
        return(model_params)

def build_params(model_config_path):
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    return(aux_build_params(model_config))
    
def get_model_constructor(name):
    if('DenseAE' in name):
        return(vae.DenseAE)
    if('DenseVAE' in name):
        return(vae.DenseVAE)
    if('ConvAE' in name):
        return(vae.ConvAE)
    if('ConvVAE' in name):
        return(vae.ConvVAE)
    if('TimeAE' in name):
        return(timevae.TimeAE)
    if('TimeVAE' in name):
        return(timevae.TimeVAE)