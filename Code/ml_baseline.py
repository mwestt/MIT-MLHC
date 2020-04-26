# ML Baseline. A script to generate datasets of placebo/placebo and placebo/drug
# trials for use with a machine learning classifier, and some benchmark 
# classifiers to assess performance on this dataset.
# Authors: Matthew West <mwest@hsph.harvard.edu>
 
from clinical_trial_generation import generate_one_trial_seizure_diaries
from endpoint_functions import calculate_MPC_p_value

from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
#from xgboost import XGBClassifier
from tqdm import tqdm
import h5py

import random
import numpy as np
import pandas as pd


def generate_ml_dataset(N=100, n_placebo=100, n_drug=100, n_base_months=2, 
                        n_maint_months=3, baseline_time_scale="weekly", 
                        maintenance_time_scale="weekly", min_seizure=4,
                        placebo_percent_effect_mean=0.1,
                        placebo_percent_effect_std_dev=0.05, 
                        drug_percent_effect_mean=0.25,
                        drug_percent_effect_std_dev=0.5,
                        save_data=None,
                        raw_counts=True):     
    """Generate structured dataset for machine learning purposes, using 
    `generate_one_trial_seizure_diaries`.
    
    Parameters
    ----------
    N : int
        Number of trials to generate. Half of them will be placebo/placebo and
        the other half placebo/drug

    n_placebo : int
        Number of placebo patients per trial.
    
    n_drug : int
        Number of drug patients per trial.
    
    n_base_months : int
        Number of months in baseline period of each arm.
    
    n_maint_months : int
        Number of months in maintenance period of each arm.
    
    baseline_time_scale : string {'daily', 'weekly'}
        Time scale on which seizure counts are generated in baseline period. 
    
    maintenance_time_scale : string {'daily', 'weekly'}
        Time scale on which seizure counts are generated in maintenance period.
    
    min_seizure : int
        Minimum number of seizures a patient can have in the baseline period.
   
    placebo_percent_effect_mean : float
        Mean of the placebo percent effect over patients.
    
    placebo_percent_effect_std_dev : float
        Standard deviation of the placebo percent effect over patients.
    
    drug_percent_effect_mean : float
        Mean of the drug percent effect over patients.

    drug_percent_effect_std_dev : float
        Standard deviation of the drug percent effect over patients.

    save_data : boolean or string
        Whether or not to save the dataset as a HDF5 file. If a string is given,
        the dataset will be named as the string. Be sure to include `.h5` as the
        file extension. Otherwise (recommended), it will be named based upon 
        salient information that went into generating the dataset, in the following 
        order: `df_{features/raw}_{N}_{n_placebo}_{n_drug}_{placebo_percent_effe
        ct_mean}_{placebo_percent_effect_std_dev}_{drug_percent_effect_mean}_{dr
        ug_percent_effect_std_dev}.h5`.
    
    raw_counts : boolean
        Whether or not to store entire raw seizure diaries in dataframe. If 
        `False`, will undergo a feature extraction step and lose individual-
        level data.

    Returns
    -------
    trial_set_df : pandas DataFrame
        Dataframe of length N, and columns for the seizure diaries of drug and
        placebo at baseline and maintenance, as well as MPC values and labels
        for drug/placebo or placebo/placebo trials.
    """

    drug_efficacy_presence = False
    data_list = []
    
    # Loop over N trials. The first N/2 will be placebo/placebo.
    print('Generating dataset of {} clinical trials.'.format(N))
    for i in tqdm(range(N)):
        if i == N / 2:
            drug_efficacy_presence = True

        p_offset = random.randint(0, 5)/25.0
        m_offset = random.randint(0, 1)/10.0
        placebo_percent_effect_mean += p_offset
        drug_percent_effect_mean += m_offset
        # Generate seizure diary for one trial
        [p_base, p_maint, t_base, t_maint] = \
         generate_one_trial_seizure_diaries(n_placebo, n_drug, n_base_months,
                                            n_maint_months,
                                            baseline_time_scale,
                                            maintenance_time_scale,
                                            min_seizure,
                                            placebo_percent_effect_mean,
                                            placebo_percent_effect_std_dev,
                                            drug_efficacy_presence,
                                            drug_percent_effect_mean,
                                            drug_percent_effect_std_dev)

        MPC_p_value = \
            calculate_MPC_p_value(baseline_time_scale, maintenance_time_scale,
                                  p_base, p_maint, t_base, t_maint)
        
        if raw_counts:
            # Append raw data to DataFrame
            data_list.append([p_base, p_maint, t_base, t_maint, MPC_p_value, 
                              int(drug_efficacy_presence)])
        
        else:                
            # Feature extraction step, defined within loop - can be tweaked

            # Define dict of raw data to call on in loop
            raw_count_dict = dict(p_base=p_base, p_maint=p_maint, t_base=t_base,
                                  t_maint=t_maint)
            
            # Feature dictionary to store generated features
            feature_dict = dict()

            # Loop over all phases of placebo and trial
            for phase in ['p_base', 't_base', 't_base', 't_maint']:
                feature_dict[phase + '_mean'] = np.mean(raw_count_dict[phase])
                feature_dict[phase + '_std'] = np.std(raw_count_dict[phase])
                feature_dict[phase + '_25_pc'] = np.percentile(raw_count_dict[phase], 25)
                feature_dict[phase + '_median'] = np.median(raw_count_dict[phase])
                feature_dict[phase + '_75_pc'] = np.percentile(raw_count_dict[phase], 75)

            # Metadata/label columns
            feature_dict['MPC'] = MPC_p_value
            feature_dict['Placebo/Drug'] = int(drug_efficacy_presence)                

            data_list.append(feature_dict.values())

    if raw_counts:
        columns = ['placebo_base', 'placebo_maint', 'drug_base', 'drug_maint', 
                   'MPC', 'Placebo/Drug']
    else:
        columns = feature_dict.keys()

    trial_set_df = pd.DataFrame(data_list, columns=columns)

    if save_data is not None:
        if isinstance(save_data, bool):
            if save_data:
                raw = 'raw' if raw_counts else 'features'
                file_name = "df_{}_{}_{}_{}_{}_{}_{}_{}.h5".format(
                    raw, N, n_placebo, n_drug, placebo_percent_effect_mean, 
                    placebo_percent_effect_std_dev, drug_percent_effect_mean, 
                    drug_percent_effect_std_dev)
                    
                trial_set_df.to_hdf(file_name, key='df')
        else:
            try:
                trial_set_df.to_hdf(save_data, key='df')
            except AttributeError as e:
                print('`save_data` must be bool or string: {}.'.format(e))

    return trial_set_df


def generate_baseline_predictions(df):
    """Function to train and generate predictions from a given dataset for 
    baseline model. Takes either a pandas DataFrame or string to HDF5 file where
    one is stored.
    
    Parameters
    ----------
    df : pandas DataFrame or string
        DataFrame of clinical trial data, or a string for the filename of a 
        `.h5` file from which to open DataFrame.

    Returns
    -------
    power : float
        Statistical power of ML method. The probability of correctly identifying
        a placebo/drug trial.

    type_1_error : float
        Type 1 error of method. The probability of incorrectly identifying a 
        placebo/placebo trial as a placebo/drug trial.
    """
    if isinstance(df, str):
        df = pd.read_hdf(df, 'df')

    # Prepare data and test/train split
    X = df.drop(columns=['MPC', 'Placebo/Drug'])    
    y = df['Placebo/Drug']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Select classifier
    lr_classifier = LogisticRegression(solver='lbfgs')
    svm_classifier = svm.SVC(gamma='scale')
    # classifier = XGBClassifier()
    rf_classifier = RandomForestClassifier(n_estimators=100)

    # Fit classifier and make predictions    
    lr_power, lr_type_1_error = predict(lr_classifier, X_train, y_train, X_test, y_test)
    svm_power, svm_type_1_error = predict(svm_classifier, X_train, y_train, X_test, y_test)
    rf_power, rf_type_1_error = predict(rf_classifier, X_train, y_train, X_test, y_test)
    print("Svm power: ", svm_power)
    print("Svm type 1 error: ", svm_type_1_error)
    print("Logistic Regression power: ", lr_power)
    print("Logistic Regression type 1 error: ", lr_type_1_error)
    print("Random forest power: ", rf_power)
    print("Random forest power: ", rf_type_1_error)

def predict(classifier, X_train, y_train, X_test, y_test):


    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    power = recall_score(y_test, y_pred)
    tn, fp, _, _ = confusion_matrix(y_test, y_pred).ravel()
    type_1_error = fp / (fp + tn)

    return power, type_1_error
if __name__ == "__main__":

    # Individual trial hyperparameters
    num_placebo_arm_patients = 100
    num_drug_arm_patients    = 100

    num_baseline_months    = 2
    num_maintenance_months = 3

    baseline_time_scale    = 'weekly'
    maintenance_time_scale = 'weekly'

    minimum_cumulative_baseline_seizure_count = 4

    placebo_percent_effect_mean    = 0.2
    placebo_percent_effect_std_dev = 0.05
    drug_percent_effect_mean       = 0.2
    drug_percent_effect_std_dev    = 0.05

    # Generate dataset - can just comment this out and use saved data
    df_dataset = generate_ml_dataset(N=2000, n_placebo=num_placebo_arm_patients,
                        n_drug=num_drug_arm_patients,
                        n_base_months=num_baseline_months, 
                        n_maint_months=num_maintenance_months,
                        baseline_time_scale=baseline_time_scale, 
                        maintenance_time_scale=maintenance_time_scale,
                        min_seizure=minimum_cumulative_baseline_seizure_count,
                        placebo_percent_effect_mean=placebo_percent_effect_mean, 
                        placebo_percent_effect_std_dev=placebo_percent_effect_std_dev, 
                        drug_percent_effect_mean=drug_percent_effect_mean, 
                        drug_percent_effect_std_dev=drug_percent_effect_std_dev,
                        save_data=True,
                        raw_counts=False)
    '''

    file = h5py.File('df_features_5000_100_100_504.6400000000105_0.05_255.99999999998974_0.05.h5', 'r', )
    df_dataset = file['df']['table'][:]
'''
    print(df_dataset.head())
    # Generate predictions
    generate_baseline_predictions(df_dataset)

    
