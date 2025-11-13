import pandas as pd
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def Random_split_with_AR(df, target_column, col_AR, parameters):
    perc_train = parameters["perc_train"]
    random_state = parameters['random_state']
    #Split features and labels
    X = df.drop(columns=target_column)
    y = df[target_column]
    #Split data to train and validation 
    if perc_train=="":
        perc_train=0.7
    if perc_train!=1:
        perc_test = 1 - perc_train
        train_x, test_x, train_y, test_y  = train_test_split(X, y, test_size=perc_test, train_size=perc_train, shuffle=True, random_state=random_state, stratify = y)
        if col_AR!="" and col_AR in X.columns:
            intersec, train_ind, test_ind = np.intersect1d(train_x[col_AR], test_x[col_AR], return_indices=True)
            if intersec.size != 0:
                for i in intersec:
                    count_train = train_x.query(col_AR+'=='+str(i))
                    count_test = test_x.query(col_AR+'=='+str(i))
                    print(count_train.shape[0], count_test.shape[0])
                    if count_train.shape[0] >= count_test.shape[0]:
                        rows = test_y.loc[count_test.index.values.tolist()]
                        train_x = pd.concat([train_x, count_test])
                        test_x = test_x.drop(count_test.index)
                        train_y = pd.concat([train_y, rows])
                        test_y = test_y.drop(rows.index)
                    else:
                        rows = train_y.loc[count_train.index.values.tolist()]
                        test_x = pd.concat([test_x, count_train])
                        train_x = train_x.drop(count_train.index)
                        test_y = pd.concat([test_y, rows])
                        train_y = train_y.drop(rows.index)
    else:
        train_x=X
        train_y=y
    txt = f"Based on the selected date range: \n - there are {train_x.shape[0]} observations in the training set ({np.round(perc_train*100,2)} %) \n - there are {test_x.shape[0]} observations in the validation set ({np.round(perc_test*100,2)} %)."
    print("I_MSG~"+txt)
    print("~MSG") 
    if col_AR=="":
        txt_noAR = f"\nThere is no column for the active region so the accuracy for the test is not garanteed"
        print("W_MSG~"+txt_noAR)
        print("~MSG") 
        txt = txt + txt_noAR
    print(txt)
    return train_x, train_y, test_x, test_y, txt

def dataset_split(training_set, df, df_group, groups, groups_rare,groups_null,harpnum, train_n):
        g_n = 0
        #--------------------------------Start to take one type at time
        for g_n in range(len(groups)):
            # print(groups[g_n])
            #--------------------------------Take the harpnums that have observation for that type
            df_harpnum_type = df_group[df_group[groups[g_n]] > 0]
            #--------------------------------Shuffle the harpnum
            df_harpnum_type_shuffled = df_harpnum_type.sample(frac=1).reset_index(drop=True)
            if groups[g_n] in groups_null:
                #--------------------------------If we want the null type try to not take rare events and keep it for the next set
                df_good = df_harpnum_type_shuffled[(df_harpnum_type_shuffled[groups_rare] == 0).all(axis=1)]
                df_other = df_harpnum_type_shuffled[(df_harpnum_type_shuffled[groups_rare] > 0).all(axis=1)]
                df_harpnum_type_shuffled = pd.concat([df_good, df_other]).reset_index(drop=True)
            harpnum_type_shuffled = df_harpnum_type_shuffled[harpnum].tolist() 
            #--------------------------------Take the right number of observation to have the right percentage of the type
            for this_harpnum in harpnum_type_shuffled:
                # Count how many rows of this type are added
                n_type = training_set[groups[g_n]].sum()
                if n_type >= train_n[groups[g_n]]:
                    n_drop = n_type - train_n[groups[g_n]]
                    if n_drop > 0:
                        all_sharp_type = training_set[training_set[groups[g_n]] >= 1]
                        df_selected = all_sharp_type.sample(n=int(n_drop))
                        training_set = training_set.drop(df_selected.index)                    
                    break
                df_selected_harp = df[(df[groups[0:g_n]] == 0).all(axis=1) & (df[harpnum] == this_harpnum)]
                if (n_type + df_selected_harp[groups[g_n]].sum() > train_n[groups[g_n]]):
                    n_add = train_n[groups[g_n]] - n_type
                    df_selected_harp = df_selected_harp[df_selected_harp[groups[g_n]]>0]
                    df_selected_harp = df_selected_harp.sample(n=int(n_add))
                training_set = pd.concat([training_set, df_selected_harp], ignore_index=True)
                df_group = df_group[df_group[harpnum] != this_harpnum]
        return training_set, df_group

def Balanced_Type_with_AR(df, groups, col_AR, parameters):
    #--------------------------------Take all the parameters usefull to do the splitting:
    #--------------------------------The columns with the flare type, the rare events, the column with the labels, the percent of the train and relax
    harpnum = col_AR
    groups_flare = parameters['flare_columns']
    groups_flare = [s.lower() for s in groups_flare]
    groups_rare = parameters['rare_events']
    groups_rare = [s.lower() for s in groups_rare]
    target_column = parameters['label_column'].lower()
    n_perc_train = parameters['perc_train']
    perc_relax = parameters['perc_relax']

    if not set(groups_flare).issubset(groups):
        print("E_MSG~"+f'The column {groups_flare} is not in the labels columns')
        print("~MSG") 
        raise ValueError(f'The column {groups_flare} is not in the labels columns')
    if not set(groups_rare).issubset(groups):
        print("E_MSG~"+f'The column {groups_rare} is not in the labels columns')
        print("~MSG") 
        raise ValueError(f'The column {groups_rare} is not in the labels columns')
    #--------------------------------Do List with the labels, all type, null type, flare type and rare flare events
    if target_column in groups:
        groups.remove(target_column)
    else: 
        print("E_MSG~"+f'The column {target_column} is not in the labels columns')
        print("~MSG") 
        raise ValueError(f'The column {target_column} is not in the labels columns')

    groups_null = [item for item in groups if item not in groups_flare] #['TYPE_NO1', 'TYPE_NO2', 'TYPE_NO3', 'TYPE_NO4']
    groups = groups_flare + groups_null
    df_tmp = df[[harpnum]+groups]
    #to remove. That command change every number grater than 1 into 1
    df[groups] = df[groups].applymap(lambda x: 1 if x > 0 else x)
    #--------------------------------Calculate the percentage of all the type in the whole dataset to can have the same rate in the train and test set 
    percent = (df[groups].sum() / df[groups].sum().sum())
    n_train_set = int(len(df) * n_perc_train * perc_relax)
    n_valid_set = int(len(df) * (1 - n_perc_train) * perc_relax)
    train_n = round(percent*n_train_set)
    valid_n = round(percent*n_valid_set)
    print("train set number \n" , train_n)
    print("valid set number \n" , valid_n)
#--------------------------------Group the dataset by the harpnum to can have the same active region in the same set
    df_group = df_tmp.groupby(by=harpnum).sum().reset_index()
    print("df group " , len(df_group))
    dataset = df.head(0).copy() #pd.DataFrame(columns=df.columns)
    training_set, df_group_post_train = dataset_split(dataset, df, df_group, groups, groups_rare,groups_null, harpnum, train_n)
    validation_set, df_group_post_valid = dataset_split(dataset, df, df_group_post_train, groups, groups_rare,groups_null,harpnum, valid_n)
#--------------------------------Verify that the test set is similar to the expectations
    tollerance_min = 0.1
    tollerance_max = 0.75
    n_try = 5
    distance_valid = (valid_n - validation_set[groups].sum())/valid_n
    for i in range(n_try):
        if distance_valid.max() >= tollerance_min and distance_valid.max() < tollerance_max:
            index_max = distance_valid.idxmax()
            n_valid_set = (validation_set[index_max].sum())/(percent[index_max])
            valid_n = round(percent*n_valid_set)
            print("percent type df \n" ,valid_n)
            g_n = 0
            for g_n in range(len(groups)):
                print(groups[g_n])
                # Count how many rows of this type are added
                n_type = validation_set[groups[g_n]].sum()
                if n_type >= valid_n[groups[g_n]]:
                    n_drop = n_type - valid_n[groups[g_n]]
                    if n_drop > 0:
                        all_sharp_type = validation_set[validation_set[groups[g_n]] >= 1]
                        df_selected = all_sharp_type.sample(n=int(n_drop))
                        validation_set = validation_set.drop(df_selected.index)
            break
        elif distance_valid.max() >= tollerance_max and i < 5:
            training_set, df_group_post_train = dataset_split(dataset, df, df_group, groups, groups_rare,groups_null, harpnum, train_n)
            validation_set, df_group_post_valid = dataset_split(dataset, df, df_group_post_train, groups, groups_rare,groups_null,harpnum, valid_n)
            i += 1
        else:
            break

    percent_type_train = (training_set[groups].sum() / training_set[groups].sum().sum())
    percent_type_valid = (validation_set[groups].sum() / validation_set[groups].sum().sum())
    percent_train = len(training_set)*100/len(df)
    percent_valid = len(validation_set)*100/len(df)
    percent_throw = 100 -(percent_train + percent_valid  )#+ percent_test
    throw_row = len(df) - (len(training_set) + len(validation_set) )#+ len(test_set)
    txt = f"Based on the selected date range: \n - there are {len(training_set)} observations in the training set ({np.round(percent_train,2)} %) \n - there are {len(validation_set)} observations in the validation set ({np.round(percent_valid,2)} %) \n - {throw_row} observations are not considered ({np.round(percent_throw,2)} %). \nWith the relax rate {perc_relax*100}%, the available observations for the splitting are {np.round(len(df)*perc_relax)}. \n"
    txt = txt + summarise(df, groups, "Original dataset")
    txt = txt + summarise(training_set,  groups, "Train")
    txt = txt + summarise(validation_set,  groups, "Test")
    #Check to verified that the AR are separeted
    harpnum_train = training_set[harpnum].tolist()
    harpnum_valid = validation_set[harpnum].tolist()
    overlap = set(harpnum_train) & set(harpnum_valid)
    if  overlap:
        print("E_MSG~"+
            "ERROR: the following HARPNUMs appear in both TRAIN and TEST "
            f"({len(overlap)}):\n  " + ", ".join(map(str, sorted(overlap))) + "\n"
        )
        print("~MSG")  
        raise ValueError('Some HARPNUMs appear in both TRAIN and TEST')
    else:
        txt = txt + "No HARPNUM appears in both TRAIN and TEST.\n"
    
    print("I_MSG~"+txt)
    print("~MSG") 
    training_set = training_set.drop(columns=groups)
    validation_set = validation_set.drop(columns=groups)

    training_set = training_set.sample(frac=1).reset_index(drop=True)
    validation_set = validation_set.sample(frac=1).reset_index(drop=True)

    train_x = training_set.drop(columns=[target_column])
    train_y = training_set[[target_column]]
    valid_x = validation_set.drop(columns=[target_column])
    valid_y = validation_set[[target_column]]
    return train_x, train_y, valid_x, valid_y, txt #, test_x, test_y

def Cronological_Split(df, target_column, col_date, parameters):
    train_periods = parameters['train_periods']
    test_periods = parameters['test_periods']
    train_dates = [tuple(p) for p in train_periods]
    test_dates = [tuple(p) for p in test_periods]
    conflicts = [
        (a, b)
        for a in train_dates
        for b in test_dates
        if not (a[1] <= b[0] or b[1] <= a[0])
    ]
    if conflicts:
        print("E_MSG~"+f"I periodi dei due elenchi si sovrappongono: {conflicts}")
        print("~MSG")  
        raise ValueError(f"I periodi dei due elenchi si sovrappongono: {conflicts}")
    merged_train = merge_periods_str(train_dates)
    merged_test = merge_periods_str(test_dates)
    df_train = filter_df_by_periods(df, col_date, merged_train)
    df_test = filter_df_by_periods(df, col_date, merged_test)
    train_x = df_train.drop(columns=target_column)
    train_y = df_train[target_column]
    valid_x = df_test.drop(columns=target_column)
    valid_y = df_test[target_column]
    return train_x, train_y, valid_x, valid_y, ''

def merge_periods(period_list):
    if not period_list:
        return []
    # Sort by start date
    period_list = sorted([(pd.to_datetime(start), pd.to_datetime(end)) for start, end in period_list], key=lambda x: x[0])
    merged = []
    current_start, current_end = period_list[0]
    for start, end in period_list[1:]:
        if start <= current_end:  # si sovrappone
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged

def merge_periods_str(period_list):
    if not period_list:
        return []
    # Ordina per start
    period_list = sorted(period_list, key=lambda x: x[0])
    merged = []
    current_start, current_end = period_list[0]
    for start, end in period_list[1:]:
        if start <= current_end:  # si sovrappone
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged

def filter_df_by_periods(df, date_col, period_list):
    if not period_list:
        return df.iloc[0:0].copy()  
    mask_list = [(df[date_col] >= start) & (df[date_col] <= end) for start, end in period_list]
    mask = pd.concat(mask_list, axis=1).any(axis=1)
    return df.loc[mask].copy()


def make_group_tables(df: pd.DataFrame, GROUP_COL, LABEL_COLS):
    y_group = (df
               .groupby(GROUP_COL)[LABEL_COLS]
               .max()
               .astype(int))

    X_group = y_group.index.to_frame(index=False)
    return X_group, y_group

# multistratify split with groupby
def stratified_group_split(Xg, yg, TEST_SIZE, RANDOM_STATE):
    msss = MultilabelStratifiedShuffleSplit(
        n_splits     = 1,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
    )
    g_train_idx, g_test_idx = next(msss.split(Xg, yg.values))

    train_groups = yg.index[g_train_idx]
    test_groups  = yg.index[g_test_idx]
    return set(train_groups), set(test_groups)

def summarise(df: pd.DataFrame, LABEL_COLS, name: str) -> str:
    lines = [f"{name:<5} — {len(df):,} rows\n"]
    for col in LABEL_COLS:
        present_rows = (df[col] > 0).sum()
        pct = present_rows / len(df) * 100 if len(df) else 0.0
        total_occ = df[col].sum()
        extra = f"; total occurrences = {int(total_occ)}" if total_occ != present_rows else ""
        lines.append(f"  {col:<8}: {present_rows:5d} rows  ({pct:5.1f} %){extra}\n")
    return "".join(lines)

def GroupBy_Stratify(df, groups, col_AR, parameters):
    GROUP_COL    = col_AR
    TEST_SIZE    = parameters['test_size']
    RANDOM_STATE = parameters['random_state']
    target_column = parameters['label_column'].lower()
    if target_column in groups:
        groups.remove(target_column)
    else: 
        print("E_MSG~"+f'The column {target_column} is not in the labels columns')
        print("~MSG") 
        raise ValueError(f'The column {target_column} is not in the labels columns')
    LABEL_COLS   = groups
    # split
    Xg, yg = make_group_tables(df, GROUP_COL, LABEL_COLS)
    train_groups, test_groups = stratified_group_split(Xg, yg, TEST_SIZE, RANDOM_STATE)
    mask_train = df[GROUP_COL].isin(train_groups)
    train_x = df.loc[mask_train].drop(columns=[target_column])
    test_x  = df.loc[~mask_train].drop(columns=[target_column])
    train_y = df.loc[mask_train, [target_column]]
    test_y  = df.loc[~mask_train, [target_column]]

    percent_train = len(train_x)*100/len(df)
    percent_test = len(test_x)*100/len(df)
    percent_throw = 100 -(percent_train + percent_test  )#+ percent_test
    throw_row = len(df) - (len(train_x) + len(test_x) )#+ len(test_set)
    report = f"Based on the selected date range: \n - there are {len(train_x)} observations in the training set ({np.round(percent_train,2)} %) \n - there are {len(test_x)} observations in the validation set ({np.round(percent_test,2)} %) \n - {throw_row} observations are not considered ({np.round(percent_throw,2)} %). \n"
    report = report + summarise(df, LABEL_COLS, "Original dataset")
    report = report + summarise(train_x, LABEL_COLS, "Train")
    report = report + summarise(test_x,  LABEL_COLS, "Test")

    train_x = train_x.drop(columns=LABEL_COLS)
    test_x = test_x.drop(columns=LABEL_COLS)
    overlap = train_groups & test_groups
    if overlap:
        print("E_MSG~"+
            "ERROR: the following HARPNUMs appear in both TRAIN and TEST "
            f"({len(overlap)}):\n  " + ", ".join(map(str, sorted(overlap))) + "\n"
        )
        print("~MSG")  
        raise ValueError('Some HARPNUMs appear in both TRAIN and TEST')
    else:
        report = report + "No HARPNUM appears in both TRAIN and TEST.\n"

    # log on terminal
    print("I_MSG~"+report)
    print("~MSG")  
    return train_x, train_y, test_x, test_y, report

def GroupBy_Stratify_without_AR(df, groups, parameters):
    TEST_SIZE    = parameters['test_size']
    RANDOM_STATE = parameters['random_state']
    target_column = parameters['label_column'].lower()
    if target_column in groups:
        groups.remove(target_column)
    else: 
        print("E_MSG~"+f'The column {target_column} is not in the labels columns')
        print("~MSG") 
        raise ValueError(f'The column {target_column} is not in the labels columns')

    X = df.drop(columns=groups)
    y = df[groups]

    y_bin = (y > 0).astype(int).values

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(msss.split(X, y_bin))

    train_x, test_x = X.iloc[train_idx], X.iloc[test_idx]
    train_type_y, test_type_y = y.iloc[train_idx], y.iloc[test_idx]
    #fare uno shuffle delle righe?
    train_y = train_x[[target_column]]
    test_y  = test_x[[target_column]]
    train_x = train_x.drop(columns=target_column)
    test_x  = test_x.drop(columns=target_column)
    percent_train = len(train_x)*100/len(df)
    percent_test = len(test_x)*100/len(df)
    percent_throw = 100 -(percent_train + percent_test  )#+ percent_test
    txt = f"Based on the selected date range: \n - there are {len(train_x)} observations in the training set ({np.round(percent_train,2)} %) \n - there are {len(test_x)} observations in the validation set ({np.round(percent_test,2)} %) \n "
    if percent_throw != 0:
        throw_row = len(df) - (len(train_x) + len(test_x) )#+ len(test_set)
        txt = txt + f"- {throw_row} observations are not considered ({np.round(percent_throw,2)} %). \n"
    txt = txt + summarise(df, groups, "Original dataset")
    txt = txt + summarise(train_type_y,  groups, "Train")
    txt = txt + summarise(test_type_y,  groups, "Test")
    print("I_MSG~"+txt)
    print("~MSG") 
    return train_x, train_y, test_x, test_y, txt