import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from itertools import compress
from scipy.stats import chi2_contingency

def chi2_table(data, target, threshold_phi=0.05):
    selected_features = []
    for feature in data.columns:
        if feature == target:
            continue
        contingency_table = pd.crosstab(data[target], data[feature])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        phi = np.sqrt(chi2 / (data.shape[0] * (min(contingency_table.shape) - 1)))
        if phi >= threshold_phi:
            selected_features.append(feature)     
    return selected_features

def driver_selection(data_set, target, threshold_r2, threshold_phi, p_value, drivers):
    num_drivers = []
    cat_drivers = []
    for driver in drivers:
        if data_set[driver].nunique() != 2:
            num_drivers.append(driver)
        else:
            cat_drivers.append(driver)

    #the target is continous        
    if data_set[target].nunique() != 2:
        #for continous variables -> Pearson correlation  
        drivers_kpi = [target] + num_drivers
        corr_matrix = data_set[drivers_kpi].corr()[0:1].iloc[:, 1:]
        final_drivers_cont = [column for column in corr_matrix.columns if any(abs(corr_matrix[column]) >= threshold_r2)]
        #for binary/categorical variables -> ANOVA
        data_set.dropna(subset=cat_drivers, inplace=True)
        X = data_set[cat_drivers]
        y = data_set[target]
        final_drivers_cat = []
        if X.empty: 
            pass
        else:
            (F,pvalues) = f_classif(X,y)
            final_drivers_cat = list(compress(data_set[cat_drivers].columns.tolist(), pvalues<=p_value))
    #the target is binary
    else:
        #for binary/categorical variables -> Phi co-efficient
        drivers_kpi = [target] + cat_drivers
        final_drivers_cat = chi2_table(data_set[drivers_kpi], target, threshold_phi)
        #for continous variables -> ANOVA
        data_set.dropna(subset=num_drivers, inplace=True)
        X = data_set[num_drivers]
        y = data_set[target]
        final_drivers_cont = [] 
        if X.empty: 
            pass
        else:
            (F,pvalues) = f_classif(X,y)
            final_drivers_cont = list(compress(data_set[num_drivers].columns.tolist(), pvalues<=p_value))
    
    final_drivers = final_drivers_cont + final_drivers_cat

    return [final_drivers, final_drivers_cont, final_drivers_cat]

def identify_multicollinearity(dataframe, threshold=0.75):
    corr_matrix = dataframe.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    
    #find features with correlation above the threshold
    correlated_features = np.where(corr_matrix > threshold)
    correlated_features = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*correlated_features) if x != y and x < y]

    unique_features = set()
    for feature_pair in correlated_features:
        unique_features.update(feature_pair)
    
    return correlated_features, list(unique_features)

def remove_mult_features(features_to_keep, correlated_features, drivers):
    for feature in features_to_keep:
        correlated_features = [e for e in correlated_features if e not in (feature)]
    
    drivers = [driver for driver in drivers if driver not in correlated_features]
    
    return drivers