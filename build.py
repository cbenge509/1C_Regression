import math
import os
#path for mingw64 (XGBoost requirement)
pathx = 'C:\\Program Files\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = pathx + ';' + os.environ['PATH']

from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingRegressor, StackingCVRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline


xgbParams = {#'n_estimators':400, 
             'n_estimators':75, 
             'n_jobs':-1, 
             'random_state':736283, 
             'learning_rate':0.5, 
             'max_depth':7, 
             'colsample_bytree':1.0, 
             'subsample':1.0,
             'gamma':0.0,
             'reg_lambda':1.0,
             'scale_pos_weight':1.0, 
             'reg_alpha':0.5}

xgbParams_new = {'n_estimators':10, 
                 'n_jobs':-1, 
                 'random_state':736283, 
                 'learning_rate':0.5, 
                 'max_depth':6, 
                 'colsample_bytree':0.85,
                 'subsample':1.0,
                 'gamma':0.0, 
                 'reg_lambda':1.0, 
                 'scale_pos_weight':1.0, 
                 'reg_alpha':0.5}

enParams = {'random_state':736283, 
            'positive':False, 
            'selection':'cyclic', 
            'l1_ratio':0.25, 
            'fit_intercept':True}

#score 0.973976 public AND 0.973926 private
lgbmParams = {'n_jobs':-1, 
             'random_state':736283, 
             'learning_rate':0.4, 
             'max_depth':10, 
             'num_leaves':70, 
             'max_bin':255, 
             'subsample_for_bin':50000,
             'subsample':1.0,
             'colsample_bytree':1.0,
             'reg_lambda':0.0001, 
             'min_child_weight':1e-3, 
             'min_child_samples':20, 
             'n_estimators':50, 
             'reg_alpha':0.0}

#score 0.986325 public AND 0.984057 private
lgbmParams_new = {'n_jobs':-1,
                  'random_state':736283,
                  'learning_rate':0.2,
                  'max_depth':10,
                  'num_leaves':70,
                  'max_bin':255,
                  'subsample_for_bin':50000,
                  'subsample':1.0,
                  'colsample_bytree':1.0,
                  'reg_lambda':0.0100,
                  'min_child_weight':1e-3,
                  'min_child_samples':20,
                  'n_estimators':60,
                  'reg_alpha':0.1}


etrParams = {'n_jobs':-1,
             'random_state':736283,
             'max_depth':13,
             'max_features':'auto',
             'min_samples_split':2,
             'min_samples_leaf':1,
             'n_estimators':50}

rfParams = {'n_jobs':-1,
            'random_state':736283,
            'max_features':0.6,
            'max_depth':None,
            'min_samples_split':2,
            'min_samples_leaf':1,
            'n_estimators':50}

adaParams = {'random_state':736283,
             'n_estimators':50}
adaDtrParams = {'random_state':736283,
                'max_depth':15,
                'max_features':None}

lrParams = {'n_jobs':-1}

mlpParams = {'random_state':736283,'hidden_layer_sizes':(100,)}

sgdParams = {'random_state':736283,'eta0':0.1,'max_iter':10}

def downcast(df):
    #reduce all float and int 64 values down to 32-bit to save memory
    floats = [c for c in df if df[c].dtype == 'float64']
    ints = [c for c in df if df[c].dtype == 'int64']
    df[floats] = df[floats].astype(np.float32)
    df[ints] = df[ints].astype(np.int32)
    
    return df
        
def build_SimpleModel():
    #model = LinearRegression(**lrParams)
    model = LGBMRegressor(**lgbmParams)
    #model = XGBRegressor(**xgbParams)
    #model = RandomForestRegressor(**rfgParams)
    #model = SGDRegressor(**sgdParams)
    #model = ElasticNet(**enParams)
    #model = ExtraTreesRegressor(**etrParams)
    #model = BaggingRegressor(DecisionTreeRegressor(random_state=736283), n_jobs=-1, random_state=736283, n_estimators=300)
    #model = MLPRegressor(**mlpParams)  #predicting at 116,184 ???
    
    #model = RandomForestRegressor(n_jobs=-1, random_state=736283, n_estimators=100)
    #model = ExtraTreesRegressor(n_jobs=1, random_state=736283, n_estimators=300)
    #model = LGBMRegressor(random_state=736283, n_estimators=700, n_jobs=-1, learning_rate=1.0, num_leaves=40, min_split_gain=0.0, min_child_samples=20,min_child_weight=5,
    #                      subsample=1.0, reg_alpha=0.009, reg_lambda=0.1,max_depth=-1,colsample_bytree=1.0,subsample_freq=1)
    stregr = model
    
    return stregr

def build_StackingModel():
    
    lgbmModel = LGBMRegressor(**lgbmParams)
    etrModel = ExtraTreesRegressor(**etrParams)
    lrModel = LinearRegression(**lrParams)
    #sgdModel = SGDRegressor(**sgdParams)
    xgbModel = XGBRegressor(**xgbParams)
    mlpModel = make_pipeline(MinMaxScaler(), MLPRegressor(**mlpParams))
    
    stregr = StackingRegressor(regressors=[lgbmModel,xgbModel, etrModel, mlpModel], meta_regressor=XGBRegressor())
    
    return stregr

def build_StackingModelCV(posBias=False, cvRun=5):
    
    xgbModel = XGBRegressor(**xgbParams)
    lgbmModel = LGBMRegressor(**lgbmParams)
    rfgModel = RandomForestRegressor(**rfParams)
    etrModel = ExtraTreesRegressor(**etrParams)
    lrModel = LinearRegression(**lrParams)
    enModel = ElasticNet(**enParams)
    mlpModel = make_pipeline(MinMaxScaler(), MLPRegressor(**mlpParams))
    adaDtrModel = AdaBoostRegressor(DecisionTreeRegressor(**adaDtrParams), **adaParams)

    meta_model = ElasticNet(positive=True, random_state=736283)
    
    if posBias:
        stregr = StackingCVRegressor(regressors=[xgbModel, lgbmModel, rfgModel, etrModel, lrModel, enModel, mlpModel, adaDtrModel], meta_regressor=meta_model, cv=cvRun)
    else:
        stregr = StackingCVRegressor(regressors=[xgbModel, lgbmModel, etrModel, mlpModel, adaDtrModel], meta_regressor=lrModel, cv=cvRun)
    return stregr