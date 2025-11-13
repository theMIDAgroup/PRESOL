import os
import numpy as np
import pandas
import time
import json
from zipfile import ZipFile
#from jsondiff import diff
import csv
import pickle
import joblib
from sklearn.feature_selection import RFE
from aux_presol34 import preprocessing_training, preprocessing_testing, metrics_classification
from aux_presol34 import HybridLogit, HybridLasso, SVC_CV, MLPClassifier_HM, \
    MLPRegressor_HM, AdaptiveLasso_CV
from aux_presol34 import RandomForest
from aux_presol34 import MultiTaskLasso_CV, AdaptiveMultiTaskLasso_CV
from aux_presol34 import MultiTaskPoissonLasso_CV, AdaptiveMultiTaskPoissonLasso_CV
import warnings; warnings.filterwarnings("ignore")


class Presol34:

    def __init__(self) -> None:

        self.ml_algorithm = {'HybridLasso': HybridLasso,
                             'HybridLogit': HybridLogit,
                             'MLPRegressor_HM': MLPRegressor_HM,
                             'MLPClassifier_HM': MLPClassifier_HM,
                             'RandomForest': RandomForest,
                             'SVC_CV': SVC_CV,
                             'MultiTaskLasso_CV': MultiTaskLasso_CV}

        df_X_training = None
        df_Y_training = None
        X_training = None
        Y_training = None

        df_X_test = None
        df_Y_test = None
        X_test = None
        Y_test = None

        # os.chdir('./algorithms')
        path_conf = './algorithms/data'
        # path_conf = '/home/valeria/Documents/work/presol_alpha/presol/data'
        config_json = open(os.path.join(path_conf, 'config.json'))
        config = json.load(config_json)
        self.parameters = config['parameters']
        
        self.std_feature = config['parameters']['preprocessing']['standardization_feature']
        self.std_label = config['parameters']['preprocessing']['standardization_label']
        
        self.algo = config['config_name']
        self.features_ranking = config["features_ranking"]

        # I parametri da qui sotto vanno chiesti all'utente da GUI
        # self.std_feature = True
        # self.std_label = False

        self.path2save = './algorithms/models_hob/'
        self.path2save_csv = './algorithms/models_hob/test_and_metrics/'
        self.path2open = './algorithms/'

    def fit(self, X_training_pkl, Y_training_pkl, n_dataset=None):

        self.df_X_training = pandas.DataFrame(X_training_pkl)
        self.df_Y_training = pandas.DataFrame(Y_training_pkl)
        df_training = self.df_X_training.copy()
        for col in self.df_X_training.columns:
            if pandas.api.types.is_string_dtype(self.df_X_training[col]):
                self.df_X_training = self.df_X_training.drop(columns=[col])
        if self.df_X_training.shape[1] == 0:
            raise Exception('CSV not Valid')
        f1 = open('./algorithms/' + 'parameters.json')
        json1 = json.load(f1)
        self.col_AR = json1['column_active_region'].lower()
        col_date = json1['selection_period']['name_column_time'].lower()
        if self.col_AR!="" and self.col_AR in self.df_X_training.columns:
            self.df_X_training = self.df_X_training.drop(columns=[self.col_AR])
        if col_date!="" and col_date in self.df_X_training.columns:
            self.df_X_training = self.df_X_training.drop(columns=[col_date])    
        self.active_features = self.df_X_training.columns
        self.labels = self.df_Y_training.columns

        self.X_training = np.array(self.df_X_training)
        self.Y_training = np.array(self.df_Y_training)

        Xn_training, Yn_training, self._mean_, self._std_, self._Ymean_, self._Ystd_ = preprocessing_training(
            self.X_training, self.Y_training,  self.std_feature, self.std_label)
        
        statistic_par = {"mean": self._mean_.tolist(), "std": self._std_.tolist(), "Ymean": self._Ymean_, "Ystd": self._Ystd_}
        json_par = json.dumps(statistic_par)
        file_par_json = self.path2save + 'json_par.json'
        with open(file_par_json, 'w') as f:
            f.write(json_par)
        
        self.classification = self.parameters[self.algo]['classification']
        self.type_class = list(self.classification.keys())[list(self.classification.values()).index(True)]
        self.filename = time.strftime("%Y%m%d-%H%M%S")+'_'+self.type_class
        if not os.path.exists(self.path2save_csv):
            os.makedirs(self.path2save_csv)
        print(self.path2save)
        print(self.filename)
        print(self.algo)

        self.file_df_training_csv = self.path2save_csv + self.filename + '_' + self.algo + '_metrics_training.csv'

        # Y predicted 
        self.file_df_Y_training_csv = self.path2save_csv + self.filename + '_' + self.algo + '_Y_predicted_training.csv' 
        
        # model
        self.file_df_model = self.path2save + self.type_class + '_' + self.algo + '_model.pkl'

        self.estimator = self.ml_algorithm[self.algo](**self.parameters)
        self.estimator.fit(Xn_training, Yn_training)
        
        features = {"feature": self.active_features.tolist(), "label":self.labels.tolist()}
        json_features = json.dumps(features)
        file_feature_json = self.path2save +'json_features.json'
        with open(file_feature_json, 'w') as f:
            f.write(json_features)

        # salva il modello su file 
        joblib.dump(self.estimator, self.file_df_model)
        #TASK 3
        threshold = None
        try:
            if type(self.estimator.threshold)==dict:
                Y_prediction = self.estimator.predict(Xn_training)
                if Yn_training.shape[1] == 1: 
                    Y_predicted_training = Y_prediction > self.estimator.threshold['0']
                    threshold = self.estimator.threshold['0']
                else:
                    Y_predicted_training = np.zeros(Y_prediction.shape)
                    for i in range(Yn_training.shape[1]):
                        Y_predicted_training[:,i] = Y_prediction[:,i] > self.estimator.threshold[str(i)]
            else:
                Y_predicted_training = self.estimator.predict(Xn_training) > self.estimator.threshold
                threshold = self.estimator.threshold
        except:
            Y_predicted_training = self.estimator.predict(Xn_training)
        
        df_training[self.df_Y_training.columns] = self.df_Y_training

        if Yn_training.shape[1] == 1:
            label_column_Y_predicted = self.df_Y_training.columns[0]+'_predicted'
            df_training[label_column_Y_predicted] = Y_predicted_training.astype(int)
        else:
            for i in range(Yn_training.shape[1]):
                label_column_Y_predicted = self.df_Y_training.columns[i]+'_predicted'
                df_training[label_column_Y_predicted] = Y_predicted_training[:,i].astype(int)

        df_training.to_csv(
            self.file_df_Y_training_csv, sep='\t', float_format='%10.4f')

        self.table_training = [Xn_training.shape[0]]
        label_column = ['#point-in-time']
        for i in range(Yn_training.shape[1]):
            self.table_training = self.table_training + [ Yn_training.sum(),
                                self.estimator.metrics_training[str(i)]['tss'],
                                self.estimator.metrics_training[str(i)]['hss'],
                                self.estimator.metrics_training[str(i)]['acc'],
                                self.estimator.metrics_training[str(i)]['far'],
                                self.estimator.metrics_training[str(i)]['fnfp'],
                                self.estimator.metrics_training[str(i)]['pod'],
                                self.estimator.metrics_training[str(i)]['balanced acc'],
                                self.estimator.metrics_training[str(i)]['balance label'] ]
            label_column = label_column + ['num_label=1', 'tss', 'hss', 'accuracy', 'far', 'fnfp', 'pod', 'balanced accuracy',
                            'balance label']
        if threshold != None:
            self.table_training = [float(threshold)] + self.table_training
            label_column = ['threshold'] + label_column

        if 'feature importance' in self.estimator.metrics_training:
            if len(self.estimator.metrics_training['feature importance']) == len(self.active_features.tolist()):
                self.table_training = self.table_training + self.estimator.metrics_training['feature importance']
                label_column = label_column + self.active_features.tolist()
            elif len(self.estimator.metrics_training['feature importance'][0]) == len(self.active_features.tolist()):
                self.table_training = self.table_training + self.estimator.metrics_training['feature importance'][0]
                label_column = label_column + self.active_features.tolist()
            
            ##########################
            # QUESTO SOTTO è IL TASK 4
            ##########################        
            if self.features_ranking:
                file_ranking_csv = self.path2save_csv + self.filename + '_' + self.algo + '_ranking.csv'
                selector = RFE(self.estimator.estimator, n_features_to_select=1)
                selector.fit(Xn_training, Yn_training)
                df_ranking = pandas.DataFrame([selector.ranking_], columns=self.active_features.tolist())
                df_ranking.to_csv(file_ranking_csv, sep='\t',float_format='%10.4f')    
            
        df_fulldata = pandas.DataFrame([self.table_training], columns=label_column)
        df_fulldata.to_csv(self.file_df_training_csv,sep='\t', float_format='%10.4f')         
        return self.estimator.metrics_training['0']['tss']
    
    def validation(self, X_val_pkl, Y_val_pkl, n_dataset=None):
        self.file_df_val_csv = self.path2save_csv + self.filename + '_' + self.algo + '_metrics_validation.csv'
        self.file_df_Y_val_csv = self.path2save_csv + self.filename + '_' + self.algo + '_Y_predicted_validation.csv'

        self.df_X_val = pandas.DataFrame(X_val_pkl)
        self.df_Y_val = pandas.DataFrame(Y_val_pkl)
        df_validation = self.df_X_val.copy()
        for col in self.df_X_val.columns:
            if pandas.api.types.is_string_dtype(self.df_X_val[col]):
                self.df_X_val = self.df_X_val.drop(columns=[col])
        if self.df_X_val.shape[1] == 0:
            raise Exception('CSV not Valid')
        if self.col_AR!="" and self.col_AR in self.df_X_val.columns:
            self.df_X_val = self.df_X_val.drop(columns=[self.col_AR])

        self.X_val = np.array(self.df_X_val)
        self.Y_val = np.array(self.df_Y_val)
        #comparing feaures test with features model
        active_features_val = self.df_X_val.columns
        print(np.all(active_features_val) == np.all(self.active_features))

        differences = (np.all(active_features_val) == np.all(self.active_features))
        if differences == False:
            raise Exception
        if  (np.all(self.df_Y_val.columns) != np.all(self.labels)):
            raise Exception

        Xn_val, Yn_val = preprocessing_testing(
            self.X_val, self._mean_, self._std_, self._Ymean_, self._Ystd_, self.std_feature, self.std_label, self.Y_val)
        
        #loads the model
        predict = self.estimator.predict(Xn_val, Yn_val)
        threshold = None
        try:
            if type(self.estimator.threshold)==dict:
                Y_prediction = self.estimator.predict(Xn_val)
                if Yn_val.shape[1] == 1: 
                    Y_predicted_val = Y_prediction > self.estimator.threshold['0']
                    threshold = self.estimator.threshold['0']
                else:
                    Y_predicted_val = np.zeros(Y_prediction.shape)
                    for i in range(Yn_val.shape[1]):
                        Y_predicted_val[:,i] = Y_prediction[:,i] > self.estimator.threshold[str(i)]
            else:
                Y_predicted_val = self.estimator.predict(Xn_val) > self.estimator.threshold
                threshold = self.estimator.threshold
        except:
            Y_predicted_val = self.estimator.predict(Xn_val)

        table_val = [Xn_val.shape[0]]
        label_column_val = ['#point-in-time']
        for i in range(Yn_val.shape[1]):
            table_val = table_val + [ Yn_val.sum(),
                        self.estimator.metrics_testing[str(i)]['tss'],
                        self.estimator.metrics_testing[str(i)]['hss'],
                        self.estimator.metrics_testing[str(i)]['acc'],
                        self.estimator.metrics_testing[str(i)]['far'],
                        self.estimator.metrics_testing[str(i)]['fnfp'],
                        self.estimator.metrics_testing[str(i)]['pod'],
                        self.estimator.metrics_testing[str(i)]['balanced acc'],
                        self.estimator.metrics_testing[str(i)]['balance label']]
            label_column_val = label_column_val + ['num_label=1', 'tss', 'hss', 'accuracy', 'far', 'fnfp', 'pod', 'balanced accuracy',
                                'balance label']

        if threshold != None:
            table_val = [float(threshold)] + table_val
            label_column_val = ['threshold'] + label_column_val
        df_val_fulldata = pandas.DataFrame([table_val], columns=label_column_val)
        df_val_fulldata.to_csv(self.file_df_val_csv, sep='\t', float_format='%10.4f')

        df_validation[self.df_Y_val.columns] = self.df_Y_val
        if Yn_val.shape[1] == 1:
            label_column_Y_predicted = self.df_Y_val.columns[0]+'_predicted' 
            df_validation[label_column_Y_predicted] = Y_predicted_val.astype(int)
        else:
            for i in range(Yn_val.shape[1]):
                label_column_Y_predicted = self.df_Y_val.columns[i]+'_predicted' 
                df_validation[label_column_Y_predicted] = Y_predicted_val[:,i].astype(int)

        df_validation.to_csv(self.file_df_Y_val_csv, sep='\t', float_format='%10.4f')
        return self.estimator.metrics_testing['0']['tss']

    def predict(self, X_test_pkl, Y_test_pkl=None, filename=""):
        path = './algorithms/'
        path2open = path + 'models_hob/'
        path2open_csv = path + 'models_hob/test_and_metrics/'
        path2save = path + 'models_hob/predict/'
        classification = self.parameters[self.algo]['classification']
        type_class = list(classification.keys())[list(classification.values()).index(True)]

        if filename == "":
            filename = time.strftime("%Y%m%d-%H%M%S")
        file_df_model = path2open + type_class + '_' + self.algo + '_model.pkl'

        self.df_X_test = pandas.DataFrame(X_test_pkl)
        self.X_test = np.array(self.df_X_test)
        if Y_test_pkl is not None:
            self.df_Y_test = pandas.DataFrame(Y_test_pkl)     
            self.Y_test = np.array(self.df_Y_test)  
        else:
            self.Y_test = None  

        #load statistic parameters 
        par_json = open(path2open + 'json_par.json')
        par = json.load(par_json)
        _mean_ = np.array(par['mean'])
        _std_ = np.array(par['std'])
        _Ymean_ = par['Ymean']
        _Ystd_ = par['Ystd']

        # testing
        Xn_test, Yn_test = preprocessing_testing(
            self.X_test, _mean_, _std_, _Ymean_, _Ystd_, self.std_feature, self.std_label, self.Y_test)
        
        #loads the model
        estimator = joblib.load(file_df_model)#to change the path to load model 
        predict = estimator.predict(Xn_test, Yn_test)
        try:
            if type(self.estimator.threshold)==dict:
                Y_prediction = estimator.predict(Xn_test)
                if Yn_test.shape[1] == 1: 
                    Y_predicted_test = Y_prediction > estimator.threshold['0']
                else:
                    Y_predicted_test = np.zeros(Y_prediction.shape)
                    for i in range(Yn_test.shape[1]):
                        Y_predicted_test[:,i] = Y_prediction[:,i] > estimator.threshold[str(i)]
            else:
                Y_predicted_test = estimator.predict(Xn_test) > estimator.threshold
        except:
            Y_predicted_test = estimator.predict(Xn_test)

        file_zip = path2save+ 'test_'+ self.algo +'.zip'
        if Y_test_pkl is not None:
            table_test = [Xn_test.shape[0]]
            label_column_testing = ['#point-in-time']
            for i in range(Yn_test.shape[1]):
                table_test = table_test + [Yn_test.sum(),
                            estimator.metrics_testing[str(i)]['tss'],
                            estimator.metrics_testing[str(i)]['hss'],
                            estimator.metrics_testing[str(i)]['acc'],
                            estimator.metrics_testing[str(i)]['far'],
                            estimator.metrics_testing[str(i)]['fnfp'],
                            estimator.metrics_testing[str(i)]['pod'],
                            estimator.metrics_testing[str(i)]['balanced acc'],
                            estimator.metrics_testing[str(i)]['balance label']]
                label_column_testing = label_column_testing + ['num_label=1', 'tss', 'hss', 'accuracy', 'far', 'fnfp', 'pod', 'balanced accuracy',
                                    'balance label']
            df_predict_fulldata = pandas.DataFrame(
                [table_test], columns=label_column_testing)
            self.file_df_test_csv = path2save + filename + '_' + self.algo + '_metrics_test.csv'
            df_predict_fulldata.to_csv(self.file_df_test_csv, sep='\t', float_format='%10.4f')
        return Y_predicted_test
