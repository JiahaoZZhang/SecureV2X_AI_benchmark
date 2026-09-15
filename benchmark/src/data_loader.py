import abench
from abench.data_loader.timeseries.data_loader import ABLoaderFromSequenceFolder,ABCVDataExperiment
from abench.data_loader.timeseries.scaler import TemporalFeatureScaler


# Exemple of a dataloder that provide (X,y),Context,metadata related to a selection
def get_DataExperiment(config_benchmark,with_test=True):
    Dataloader_Train = ABLoaderFromSequenceFolder(path='data/',
                               depth_name_list = ['dataset','set'],
                               constraint_selection_list = [('dataset',config_benchmark['datasets']),('set',config_benchmark['sets_norm_scale'])],
                               constraint_rejection_list = [],
                               w_size=50,
                               sampling=1, 
                               sample_stride=50,
                               horizon_start=0,
                               prediction_number=50,
                               y_step=1,
                               x_features= ['positionX', 'positionY', 'VelX', 'VelY'],
                               y_features = ['positionX', 'positionY', 'VelX','VelY'],
                               context_features = config_benchmark['context_features'],
                               with_context=True,
                               with_metadata=True,
                               Xscaler = TemporalFeatureScaler(),
                               Yscaler = TemporalFeatureScaler(),
                               shuffle=False,
                               name='set_2&3',
                               dir_cache=None)
    
    for (X,y),context,metadata in Dataloader_Train:
        pass
    
    Xscaler_train = Dataloader_Train.Xscaler
    Yscaler_train = Dataloader_Train.Yscaler
    
    ABloader_dict_params = {'path':'data/',
                            'depth_name_list':['dataset','set'],
                            'constraint_selection_list':[('dataset',config_benchmark['datasets'])],
                            'constraint_rejection_list':[],
                            'w_size':50,'sampling':1,'sample_stride':50,'horizon_start':0,'prediction_number':50,'y_step':1,
                            'x_features':['positionX', 'positionY', 'VelX', 'VelY'],
                            'y_features':['positionX', 'positionY', 'VelX','VelY'],
                            'context_features':config_benchmark['context_features'],
                            'with_context':True,'with_metadata':True,
                            'Xscaler':Xscaler_train, 'Yscaler':Yscaler_train,
                            'name':'data'}


    
    # DataExperimentPlan : Cross validation     
    DataExperiment = ABCVDataExperiment(ABLoaderFromSequenceFolder,ABloader_dict_params,depth_name='set',subjet_ids=config_benchmark['sets_cv_exp'],validation_config=config_benchmark['validation_config'],with_test=with_test)
    DataExperiment.check_set_names()
    return(DataExperiment)