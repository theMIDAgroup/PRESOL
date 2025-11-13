# place code here
import psycopg2
import sys
import os
import glob
import ast
import json
from pandas import read_sql_query
import pandas as pd
import main_presol34 as presol
from collections import Counter

path = './algorithms/'
path2open = path + 'models_hob/'
path2open_csv = path + 'models_hob/test_and_metrics/'
path2save = path + 'models_hob/predict/'
if not os.path.exists(path2save):
            os.makedirs(path2save)
prediction_id = '45721181-b727-4312-9f4f-da0b11247df7' #'67f4a194-1d97-49e6-ab32-702e73e42799' #sys.argv[3]  
dataset = pd.read_csv('./dataset/final_dataset.csv')
if dataset.columns.duplicated().any():
    Exception('Ci sono colonne con il medesimo nome')
# True se ci sono nomi duplicati
#--------------------------------Open the file json with list of features and labels saved during the training process
with open(path2open+'json_features.json') as f1:
    json1 = json.load(f1)
features = json1['feature']
# features = [s.lower() for s in features] #da eliminare
labels = json1['label']
#--------------------------------Open file json with the ar and date columns filled by the user during the training process
f1 = open(path + 'parameters.json')
json1 = json.load(f1)
col_AR = json1['column_active_region'].lower()
col_date = json1['selection_period']['name_column_time'].lower()
#---da riflettere come sistemare il json, se mettiamo anche la colonna labels, allora quella potrebbe essere il target e le altre le togliamo semplicemente 
# if labels!=target_column:
#     raise Exception('Problem with the training saving problem') 
obj = presol.Presol34()
#--------------------------------Check if the label column exists in the dataset and separate it in case
if pd.Series(labels).isin(dataset.columns).any():
    df_x_test = dataset.drop(columns=labels)
    df_Y = dataset[labels]

    df_y_test = pd.DataFrame(columns=labels)
    for col in labels:
        if pd.api.types.is_string_dtype(df_Y[col]):
            try:
                df_y_test[col] = df_Y[col].astype(float)
            except:
                raise Exception('Labels in csv are not Valid')
    labels_test = df_Y.columns.tolist()
    if labels!=labels_test:
        raise Exception('Labels in csv are not valid') 
else:
    df_x_test = dataset
    df_y_test = None

df_test = df_x_test.copy()
txt = ''
#--------------------------------Check if some observations are present in the training set
cols = [col_AR, col_date]
if col_AR!="" and col_date!="":
    pattern = os.path.join(path2open_csv, "*_Y_predicted_training.csv")
    file_list = glob.glob(pattern)
    if not file_list:
        raise Exception('Problem with the training process') 
    else:
        file_path = file_list[0]  
        print(f"Open: {file_path}")
        df_training = pd.read_csv(file_path, sep='\t')
    
    has_all = set(cols).issubset(df_x_test.columns)
    if has_all:
        for col in cols:
            if pd.api.types.is_string_dtype(df_x_test[col]):
                try:
                    df_x_test[col] = df_x_test[col].astype(float)
                except:
                    df_x_test[col] = df_x_test[col]
        key_training = [tuple(x) for x in df_training[cols].to_numpy()]
        key_test = [tuple(x) for x in df_x_test[cols].to_numpy()]
        counts_train = Counter(key_training)
        counts_test = Counter(key_test)
        remaining = {k: max(0, counts_test[k] - counts_train.get(k, 0)) for k in counts_test}
        mask_new = []
        for k in key_test:
            if remaining.get(k, 0) > 0:
                mask_new.append(True)
                remaining[k] -= 1
            else:
                mask_new.append(False)
        if any(mask_new):
            txt_common = 'The provided dataset has observations in common with the training dataset. The prediction will be performed both for the entire provided dataset and for the new observations only.'
            print(txt_common)
            txt = txt + txt_common
            df_x_new = df_x_test.loc[mask_new].copy()
            df_test_new = df_x_new.copy()
            if df_y_test is not None:
                df_y_new = df_y_test.loc[mask_new].copy()
        else:
            print("the dataset is new")
            df_x_new = None
    else:
        df_x_new = None
        print('dataset non ha le colonne')
else:
    df_x_new = None
    txt_no_ar = f"Since no column corresponding to active regions was provided during training, the reliability of the predictions cannot be guaranteed. \n"
    print(txt_no_ar)
    txt = txt + txt_no_ar
#--------------------------------Remove Ar column and date column
list_col = []
for s in cols:
    if s.strip():  
        list_col.append(s)
if len(list_col)>0 and set(list_col).issubset(df_x_test.columns):
    df_x_test = df_x_test.drop(columns=list_col)
    if df_x_new is not None:
        df_x_new = df_x_new.drop(columns=list_col)
#--------------------------------Transform the data in float
for col in df_x_test.columns:
    if pd.api.types.is_string_dtype(df_x_test[col]):
        try:
            df_x_test[col] = df_x_test[col].astype(float)
            if df_x_new is not None:
                df_x_new[col] = df_x_new[col].astype(float)
        except:
            df_x_test = df_x_test.drop(columns=[col])
            if df_x_new is not None:
                df_x_new = df_x_new.drop(columns=[col])
if df_x_test.shape[1] == 0:
    raise Exception('The data in the CSV are not valid')
#--------------------------------Check if there are all the necessary features
active_features = df_x_test.columns.tolist()
presence_feature = all(x in active_features for x in features)
lacking_features = [x for x in features if x not in active_features]
extra_features = [x for x in active_features if x not in features]
if presence_feature:
    if features != active_features:
        df_x_test = df_x_test[features]
        df_x_new = df_x_new[features]
    txt_features = 'The necessary features are present, so predictions can be made.\n'  
    if len(extra_features)>=1:
        txt_features = txt_features + f'There are some extra features, in particular {extra_features}.\n'
    print(txt)
    txt = txt + txt_features
else:
    txt_features = f'Some features are missing so it is not possible to continue. In particular, {lacking_features} are missing.\n'
    if len(extra_features)>=1:
        txt_features = txt_features + f'There are some extra features, in particular {extra_features}.\n'
    raise Exception(txt_features)
#---------------------------------Check if at the end the features of the test set are the same of the training set
features_test = df_x_test.columns.tolist()
features_test_new = df_x_new.columns.tolist()
if features!=features_test:
    raise Exception('The format of data of the features is not valid')
if features!=features_test_new:
    raise Exception('The format of new data of the features is not valid')
#---------------------------------Save the message 
with open(path2save + "./message_prediction.txt", "w") as file:
    file.write(txt)
#---------------------------------Test on new observations and save it
if df_x_new is not None:
    if df_y_test is not None:
        y_predicted_new = obj.predict(df_x_new, df_y_new, 'partial')
        df_test_new[df_y_new.columns] = df_y_new
    else:
        y_predicted_new = obj.predict(df_x_new, 'partial')
    
    if len(labels) == 1:
        label_column_Y_predicted_new = labels[0]+'_predicted' 
        df_test_new[label_column_Y_predicted_new] = y_predicted_new.astype(int)
    else:
        for i in range(y_predicted_new.shape[1]):
            label_column_Y_predicted_new = labels[i]+'_predicted' 
            df_test_new[label_column_Y_predicted_new] = y_predicted_new[:,i].astype(int)
    
    csv_new_filename = path2save+ 'test_new_observations_prediction.csv'
    df_test_new.to_csv(csv_new_filename, index=False, sep='\t', float_format='%10.4f')
#---------------------------------Test on all observations and save it
if df_y_test is not None:
    y_predicted = obj.predict(df_x_test, df_y_test, 'complete')
    df_test[df_y_test.columns] = df_y_test
else:
    y_predicted = obj.predict(df_x_test)

if len(labels) == 1:
    label_column_Y_predicted = labels[0]+'_predicted' 
    df_test[label_column_Y_predicted] = y_predicted.astype(int)
else:
    for i in range(y_predicted.shape[1]):
        label_column_Y_predicted = labels[i]+'_predicted' 
        df_test[label_column_Y_predicted] = y_predicted[:,i].astype(int)

csv_filename = path2save+ 'test_prediction.csv'
df_test.to_csv(csv_filename, index=False, sep='\t', float_format='%10.4f')

    
# con = psycopg2.connect(con_str)  
# qry_epoch_log = "UPDATE predictions SET status = 'D' where prediction_id = '" + prediction_id + "'"
# cur_epochs_log = con.cursor()
# cur_epochs_log.execute(qry_epoch_log)      
# con.commit()
# con.close()

print("All done :-)")

