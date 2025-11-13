from torch.utils.data import Dataset, random_split
import psycopg2
from pandas import read_sql_query
import pandas as pd
import re
import os
import ast
import numpy as np
import json
import main_presol34 as presol
import splitting
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings("ignore")


#--------------------------------Take the input data         

training_id = 'training_id' 
model_name = "A_model_prova"
# If you want to use your own datset, you have to set the following information
df = pd.read_csv('./dataset/final_dataset.csv') #Open a dataframe with features and labels together  
target_column = ['flaring'] #Do a list with the label column names
print(df.columns)
#--------------------------------Read from the json 'parameters' to get the harpnum, date and information to do the splitting
with open('./algorithms/parameters.json') as f1:
            json1 = json.load(f1)
col_date = json1['selection_period']['name_column_time'].lower()
date1 = json1['selection_period']['start']
date2 = json1['selection_period']['end']
split_name = json1['split_name']
col_AR = json1['column_active_region'].lower() 
parameters = json1['parameters'][split_name]
if (col_date!="" and col_date not in df.columns) or (col_AR!="" and col_AR not in df.columns):
    print("E_MSG~"+'The date column or active region column is not in the csv')
    print("~MSG")
    raise Exception('The date column or active region column is not in the csv')
#--------------------------------Select the period that is chosen
try: 
    if date1!="" and date2!="":
        df = df[(df[col_date]>min(date1, date2)) & (df[col_date]<max(date1, date2))]
    elif date1!=""and date2=="":
        df = df[(df[col_date]>(date1))]
    elif date1==""and date2!="":
        df = df[(df[col_date]<(date2))]
    if df.shape[0] == 0:
        print("E_MSG~"+'With the selected dates, the dataset becomes empty')
        print("~MSG")
        raise Exception('With the selected dates, the dataset becomes empty')
except:
        print("E_MSG~"+'The date format is not acceptable')
        print("~MSG")
        raise Exception('The date format is not acceptable')
#--------------------------------Trasform the data in float
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col]):
        try:
            df[col] = df[col].astype(float)
        except:
            df[col]= df[col] #df.drop(columns=[col])
if pd.api.types.infer_dtype(df.values) == 'string':
    print("E_MSG~"+'Csv is not valid')
    print("~MSG")
    raise Exception('Csv is not valid')
#--------------------------------Get the train and test set after the chosen split
if split_name == "Random_without_Type":
    train_x, train_y, test_x, test_y, txt= splitting.Random_split_with_AR(df, target_column, col_AR, parameters)
elif split_name == "Balanced_Type" and col_AR!="":
    train_x, train_y, test_x, test_y, txt = splitting.Balanced_Type_with_AR(df, target_column, col_AR, parameters)
elif split_name == "Cronological_Split":
    train_x, train_y, test_x, test_y, txt = splitting.Cronological_Split(df, target_column, col_date, parameters)
elif split_name == "GroupBy_Stratify":
    train_x, train_y, test_x, test_y, txt = splitting.GroupBy_Stratify(df, target_column, col_AR, parameters)
elif split_name == "GroupBy_Stratify_without_AR":
    train_x, train_y, test_x, test_y, txt = splitting.GroupBy_Stratify_without_AR(df, target_column, parameters)
else:
    print("E_MSG~"+'The chosen Splitting does not exist')
    print("~MSG")  
    raise ValueError('The chosen Splitting does not exist')
with open("./algorithms/models_hob/message_training.txt", "w") as file:
    file.write(txt)
#--------------------------------Perform the train and test
obj = presol.Presol34()
acc_val = obj.fit(train_x, train_y)#, training_id, model_name
if  test_x.shape[0]!=0:
    acc_val = obj.validation(test_x, test_y) #, training_id, model_name
acc = str(np.round(acc_val,4))

# df_y_test = df1[target_column]
# for col in df_y_test.columns:
#     if pd.api.types.is_string_dtype(df_y_test[col]):
#         try:
#             df_y_test[col] = df_y_test[col].astype(float)
#         except:
#             raise Exception('Labels in csv not Valid')
            
# obj.predict(test_x, training_id, 'ciao',test_y)