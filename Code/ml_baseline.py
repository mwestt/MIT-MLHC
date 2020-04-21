# ML Baseline. A script to generate sets of placebo/placebo and placebo/drug
# trials for use with a machine learning classifier, and some benchmark 
# classifiers to assess performance on this dataset.
# Authors: Matthew West <mwest@hsph.harvard.edu>
 

from endpoint_functions import calculate_MPC_p_value
from clinical_trial_generation import generate_one_trial_seizure_diaries
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
import pandas as pd


def generate_ml_dataset(N=100, n_placebo=100, n_drug=100, n_base_months=2, 
                        n_maint_months=3, baseline_time_scale="weekly", 
                        maintenance_time_scale="weekly", min_seizure=4,
                        placebo_percent_effect_mean=0.1,
                        placebo_percent_effect_std_dev=0.05, 
                        drug_percent_effect_mean=0.25,
                        drug_percent_effect_std_dev=0.5,
                        save_data=None):     
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
        file extension.
    
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
    for i in range(N):
        if i == N / 2:
            drug_efficacy_presence = True

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
        
        data_list.append([np.mean(p_base), np.mean(p_maint), np.mean(t_base), 
                          np.mean(t_maint), MPC_p_value, 
                          int(drug_efficacy_presence)])

    trial_set_df = pd.DataFrame(data_list, columns=['placebo_base', 'placebo_maint', 
                                            'drug_base', 'drug_maint', 'MPC', 
                                            'Placebo/Drug'])
    
    if save_data is not None:
        if isinstance(save_data, bool):
            if save_data:
                trial_set_df.to_hdf('trial_dataset.h5', key='df')
        else:
            try:
                trial_set_df.to_hdf(save_data, key='df')
            except AttributeError as e:
                print('`save_data` must be bool or string: {}.'.format(e))

    return trial_set_df


def generate_baseline_predictions(df):
    """Script to train and generate predictions from a given dataset for 
    baseline model.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame of clinical trial data.

    Returns
    -------
    power : float
        Statistical power of ML method. The probability of correctly identifying
        a placebo/drug trial.
    type_1_error : float
        Type 1 error of method. The probability of incorrectly identifying a 
        placebo/placebo trial as a placebo/drug trial.
    """
    X = df[['placebo_base', 'placebo_maint', 'drug_base', 'drug_maint']]
    y = df['Placebo/Drug']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # classifier = svm.SVC()
    classifier = XGBClassifier()
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(accuracy_score(y_test, y_pred))


    #return power, type_1_error


if __name__ == "__main__":

    num_placebo_arm_patients = 100
    num_drug_arm_patients    = 100

    num_baseline_months    = 2
    num_maintenance_months = 3

    baseline_time_scale    = 'weekly'
    maintenance_time_scale = 'weekly'

    minimum_cumulative_baseline_seizure_count = 4

    placebo_percent_effect_mean    = 0.1
    placebo_percent_effect_std_dev = 0.05
    drug_percent_effect_mean       = 0.25
    drug_percent_effect_std_dev    = 0.05


    df_dataset = generate_ml_dataset(N=100, n_placebo=num_placebo_arm_patients, 
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
                        save_data=False)
    
    generate_baseline_predictions(df_dataset)
    
