# ML Baseline. A script to generate datasets of placebo/placebo and placebo/drug
# trials for use with a machine learning classifier, and some benchmark 
# classifiers to assess performance on this dataset.
# Authors: Matthew West <mwest@hsph.harvard.edu>
 
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

from clinical_trial_generation import generate_one_trial_seizure_diaries
from endpoint_functions import calculate_MPC_p_value

from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from xgboost import XGBClassifier, Booster
from tqdm import tqdm

# Nice plot parameters
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


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
        the other half placebo/drug.

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
        if i == N // 2:
            drug_efficacy_presence = True

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
                feature_dict[phase + '_median'] = np.median(raw_count_dict[phase])
                feature_dict[phase + '_25_pc'] = np.percentile(raw_count_dict[phase], 25)
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


def generate_ml_dataset_large(N=100000,
                              n_base_months=2, 
                              n_maint_months=3, 
                              baseline_time_scale="weekly", 
                              maintenance_time_scale="weekly", 
                              min_seizure=4,
                              save_data=None,
                              raw_counts=True):     
    """Generate structured dataset for machine learning purposes, using 
    `generate_one_trial_seizure_diaries` for a range of trial sizes, drug
    effects, and placebo effects.
    
    Parameters
    ----------
    N : int
        Number of trials to generate. Half of them will be placebo/placebo and
        the other half placebo/drug.

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
   
    save_data : boolean or string
        Whether or not to save the dataset as a HDF5 file. If a string is given,
        the dataset will be named as the string. Be sure to include `.h5` as the
        file extension. Otherwise (recommended), it will be named based upon 
        salient information that went into generating the dataset, in the 
        following order: `df_{features/raw}_{N}.h5`.
    
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
        if i == N // 2:
            drug_efficacy_presence = True

        # Draw patient numbers and drug effect parameters randomly
        n_patients = np.random.randint(10, 500)
        placebo_percent_effect_mean = np.random.uniform(0, 0.4)
        placebo_percent_effect_std_dev = np.random.uniform(0, 0.1)
        drug_percent_effect_mean = np.random.uniform(0, 0.4)
        drug_percent_effect_std_dev = np.random.uniform(0, 0.1)

        # Generate seizure diary for one trial
        [p_base, p_maint, t_base, t_maint] = \
         generate_one_trial_seizure_diaries(n_patients, n_patients, 
                                            n_base_months,
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
                              n_patients, int(drug_efficacy_presence)])
        
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
            feature_dict['Patients per arm'] = n_patients

            data_list.append(feature_dict.values())

    if raw_counts:
        columns = ['placebo_base', 'placebo_maint', 'drug_base', 'drug_maint', 
                   'MPC', 'n_patients', 'Placebo/Drug']
    else:
        columns = feature_dict.keys()

    trial_set_df = pd.DataFrame(data_list, columns=columns)

    if save_data is not None:
        if isinstance(save_data, bool):
            if save_data:
                raw = 'raw' if raw_counts else 'features'
                file_name = "df_{}_{}.h5".format(raw, N)
                trial_set_df.to_hdf(file_name, key='df')
        else:
            try:
                trial_set_df.to_hdf(save_data, key='df')
            except AttributeError as e:
                print('`save_data` must be bool or string: {}.'.format(e))

    return trial_set_df


def generate_baseline_predictions(df, classifier_type='xgboost', 
                                  MPC_significance=0.05, save_model=False):
    """Function to train and generate predictions from a given dataset for 
    baseline model. Takes either a pandas DataFrame or string to HDF5 file where
    one is stored. Can also save trained classifier by pickling. 
    
    Parameters
    ----------
    df : pandas DataFrame or string
        DataFrame of clinical trial data, or a string for the filename of a 
        `.h5` file from which to open DataFrame.
    
    classifier_type : string {'logistic', 'svm', 'xgboost'}
        Which classifier to use to make predictions.
        
    MPC_significance : float
        Significance level for MPC value below which the null hypothesis is 
        rejected and a trial is determined to have drug effect present. 

    save_model : boolean
        Whether or not to save model. Recommended only for 'xgboost' classifier
        and when `df` is a string and thus being loaded from a stored file. 

    Returns
    -------
    power : float
        Statistical power of ML method. The probability of correctly identifying
        a placebo/drug trial.

    type_1_error : float
        Type 1 error of method. The probability of incorrectly identifying a 
        placebo/placebo trial as a placebo/drug trial.

    mpc_power : float
        Statistical power of MPC method. The probability of correctly 
        identifying a placebo/drug trial.

    mpc_type_1_error : float
        Type 1 error of MPC method. The probability of incorrectly identifying a 
        placebo/placebo trial as a placebo/drug trial.
    """
    if isinstance(df, str):
        df_file_name = df
        df = pd.read_hdf(df, 'df')
        
    # Get MPC predictions
    df['MPC_pred'] = (df['MPC'] < MPC_significance)

    # Prepare data and test/train split
    
    df_train, df_test = train_test_split(df, test_size=0.25)

    X_train = df_train.drop(columns=['MPC', 'MPC_pred', 'Placebo/Drug'])    
    X_test = df_test.drop(columns=['MPC', 'MPC_pred', 'Placebo/Drug'])
    y_train = df_train['Placebo/Drug']
    y_test = df_test['Placebo/Drug']

    # Select classifier
    if classifier_type == 'logistic':
        classifier = LogisticRegression()
    elif classifier_type == 'svm':
        classifier = svm.SVC()
    elif classifier_type == 'xgboost':
        classifier = XGBClassifier()
    else: 
        raise ValueError(
            "classifier_type must be one of {'logistic', 'svm', 'xgboost'}."
            )

    # Fit classifier and make predictions    
    classifier.fit(X_train, y_train)

    if save_model:
        try:
            pickle.dump(classifier, open(classifier_type + '_model_' + df_file_name + '.model', 'wb'))
        except UnboundLocalError as unb:
            print('Warning:', unb, 'saving with generic name')
            pickle.dump(classifier, open(classifier_type + '_model.model', 'wb'))            

    y_pred = classifier.predict(X_test)

    power = recall_score(y_test, y_pred)
    tn, fp, _, _ = confusion_matrix(y_test, y_pred).ravel()
    type_1_error = fp / (fp + tn)

    # Store predictions from MPC    
    mpc_pred = df_test['MPC_pred']

    # MPC power and type 1 error
    mpc_power = recall_score(y_test, mpc_pred)
    tn_mpc, fp_mpc, _, _ = confusion_matrix(y_test, mpc_pred).ravel()
    mpc_type_1_error = fp_mpc / (fp_mpc + tn_mpc)

    return power, type_1_error, mpc_power, mpc_type_1_error


def generate_power_from_classifier(df, 
                                         classifier=None, 
                                         MPC_significance=0.05):
    """Function to generate estimates of power and type 1 error from a given 
    dataset for a given classifier on model, WITHOUT training first. Also 
    generates these estimates for MPC method on the same data.
    
    Takes either a pandas DataFrame or string to HDF5 file where one is stored.
    
    Parameters
    ----------
    df : pandas DataFrame or string
        DataFrame of clinical trial data for making predictions on, or a string 
        for the filename of a `.h5` file from which to open DataFrame.
    
    classifier : string {'logistic', 'svm', 'xgboost'}
        Which classifier to use to make predictions.
        
    MPC_significance : float
        Significance level for MPC value below which the null hypothesis is 
        rejected and a trial is determined to have drug effect present. 

    Returns
    -------
    power : float
        Statistical power of ML method. The probability of correctly identifying
        a placebo/drug trial.

    type_1_error : float
        Type 1 error of method. The probability of incorrectly identifying a 
        placebo/placebo trial as a placebo/drug trial.

    mpc_power : float
        Statistical power of MPC method. The probability of correctly 
        identifying a placebo/drug trial.

    mpc_type_1_error : float
        Type 1 error of MPC method. The probability of incorrectly identifying a 
        placebo/placebo trial as a placebo/drug trial.
    """
    if isinstance(df, str):
        df = pd.read_hdf(df, 'df')
    
    # Get MPC predictions
    mpc_pred = (df['MPC'] < MPC_significance)

    X = df.drop(columns=['MPC', 'Placebo/Drug'])    
    y = df['Placebo/Drug']
 
    y_pred = classifier.predict(X)

    power = recall_score(y, y_pred)
    tn, fp, _, _ = confusion_matrix(y, y_pred).ravel()
    type_1_error = fp / (fp + tn)
 
    # MPC power and type 1 error
    mpc_power = recall_score(y, mpc_pred)
    tn_mpc, fp_mpc, _, _ = confusion_matrix(y, mpc_pred).ravel()
    mpc_type_1_error = fp_mpc / (fp_mpc + tn_mpc)

    return power, type_1_error, mpc_power, mpc_type_1_error


def plot_drug_effect_power_curve(drug_effects=None, N=500,
                                 classifier_type='xgboost'):
    """A function to plot power and type 1 error curves for a given dataset for
    benchmark classifiers as a function of drug effect.
    
    Parameters
    ----------
    drug_effects : array-like
        List or array containing range of mean drug effects. 
    
    N : int
        Number of clinical trials in each dataset.

    classifier_type : string {'logistic', 'svm', 'xgboost'}
        Classifier to train for each drug effect.
    """ 
    if drug_effects is None:
        drug_effects = np.linspace(0, 0.4, 41)
    
    power_list, type_1_error_list = [], []
    mpc_power_list, mpc_type_1_error_list = [], []

    for drug_percent_effect_mean in tqdm(drug_effects):
        df_dataset = generate_ml_dataset(N=N, n_placebo=100, 
                                         n_drug=100,
                                         n_base_months=2, 
                                         n_maint_months=3,
                                         baseline_time_scale='weekly', 
                                         maintenance_time_scale='weekly',
                                         min_seizure=4,
                                         placebo_percent_effect_mean=0.1, 
                                         placebo_percent_effect_std_dev=0.05, 
                                         drug_percent_effect_mean=drug_percent_effect_mean, 
                                         drug_percent_effect_std_dev=0.05,
                                         save_data=False,
                                         raw_counts=False)
    
        # Generate predictions
        power, type_1_error, mpc_power, mpc_type_1_error = \
            generate_baseline_predictions(df_dataset, 
                                          classifier_type=classifier_type)

        power_list.append(power)
        type_1_error_list.append(type_1_error)
        mpc_power_list.append(mpc_power)
        mpc_type_1_error_list.append(mpc_type_1_error)

    plt.plot(drug_effects, power_list, label='ML Power')
    plt.plot(drug_effects, type_1_error_list, color='r', label='ML Type 1 Error')
    plt.plot(drug_effects, mpc_power_list, ls='--', label='MPC Power')
    plt.plot(drug_effects, mpc_type_1_error_list, ls='--', label='MPC Type 1 Error')
    plt.axhline(y=0.9, color='r', lw=0.5, ls='-.')
    plt.axhline(y=0.05, color='r', lw=0.5, ls='-.')

    plt.xlabel('Mean Drug Effect')
    plt.ylabel('Performance')
    plt.legend()
    plt.show()


def plot_n_patient_power_curve(patient_numbers=None, N=500,
                               classifier_type='xgboost'):
    """A function to plot power and type 1 error curves for a given dataset for
    benchmark classifiers as a function of trial size.
    
    Parameters
    ----------
    patient_numbers : array-like
        List or array containing range of patient numbers 
    
    N : int
        Number of clinical trials in each dataset.

    classifier_type : string {'logistic', 'svm', 'xgboost'}
        Classifier to train for each drug effect.
    """ 
    if patient_numbers is None:
        patient_numbers = np.arange(20, 500, 50)
    
    power_list, type_1_error_list = [], []
    mpc_power_list, mpc_type_1_error_list = [], []

    for n_patients in tqdm(patient_numbers):
        df_dataset = generate_ml_dataset(N=N, n_placebo=n_patients, 
                                         n_drug=n_patients,
                                         n_base_months=2, 
                                         n_maint_months=3,
                                         baseline_time_scale='weekly', 
                                         maintenance_time_scale='weekly',
                                         min_seizure=4,
                                         placebo_percent_effect_mean=0.21, 
                                         placebo_percent_effect_std_dev=0.1, 
                                         drug_percent_effect_mean=0.2, 
                                         drug_percent_effect_std_dev=0.05,
                                         save_data=False,
                                         raw_counts=False)
    
        # Generate predictions
        power, type_1_error, mpc_power, mpc_type_1_error = \
            generate_baseline_predictions(df_dataset, 
                                          classifier_type=classifier_type)

        power_list.append(power)
        type_1_error_list.append(type_1_error)
        mpc_power_list.append(mpc_power)
        mpc_type_1_error_list.append(mpc_type_1_error)

    plt.plot(patient_numbers, power_list, label='ML Power')
    plt.plot(patient_numbers, type_1_error_list, color='r', label='ML Type 1 Error')
    plt.plot(patient_numbers, mpc_power_list, ls='--', label='MPC Power')
    plt.plot(patient_numbers, mpc_type_1_error_list, ls='--', label='MPC Type 1 Error')
    plt.axhline(y=0.9, color='r', lw=0.5, ls='-.')
    plt.axhline(y=0.05, color='r', lw=0.5, ls='-.')

    plt.xlabel('Patient Number Per Trial Arm')
    plt.ylabel('Performance')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # Individual trial hyperparameters
    num_placebo_arm_patients = 100
    num_drug_arm_patients    = 100

    num_baseline_months    = 2
    num_maintenance_months = 3

    baseline_time_scale    = 'weekly'
    maintenance_time_scale = 'weekly'

    minimum_cumulative_baseline_seizure_count = 4

    placebo_percent_effect_mean    = 0.21
    placebo_percent_effect_std_dev = 0.1
    drug_percent_effect_mean       = 0.2
    drug_percent_effect_std_dev    = 0.05

    # # Generate dataset - can just comment this out and use saved data
    # df_dataset = generate_ml_dataset(N=500, n_placebo=num_placebo_arm_patients, 
    #                     n_drug=num_drug_arm_patients,
    #                     n_base_months=num_baseline_months, 
    #                     n_maint_months=num_maintenance_months,
    #                     baseline_time_scale=baseline_time_scale, 
    #                     maintenance_time_scale=maintenance_time_scale,
    #                     min_seizure=minimum_cumulative_baseline_seizure_count,
    #                     placebo_percent_effect_mean=placebo_percent_effect_mean, 
    #                     placebo_percent_effect_std_dev=placebo_percent_effect_std_dev, 
    #                     drug_percent_effect_mean=drug_percent_effect_mean, 
    #                     drug_percent_effect_std_dev=drug_percent_effect_std_dev,
    #                     save_data=False,
    #                     raw_counts=False)
    
    # # Generate predictions
    # power, type_1_error = generate_baseline_predictions(df_dataset, 
    #                                                     classifier_type='xgboost')
    # print('Power = ', power)
    # print('Type 1 Error = ', type_1_error)

    # Generate plot of power and type 1 error 
    # plot_drug_effect_power_curve(drug_effects=np.linspace(0, 0.25, 26), N=5000)
    plot_n_patient_power_curve(patient_numbers=np.arange(10, 350, 10), N=6000)
    # generate_ml_dataset_large(N=1000, save_data=True, raw_counts=False)
    # print(generate_baseline_predictions('df_features_100000.h5', 
    #                                     classifier_type='xgboost', save_model=False))
    
    # # Loading model and using it to get power and type 1 error
    # classif = pickle.load(open("xgboost_model_df_features_100000.h5.model", "rb"))
    # print(generate_power_from_classifier('df_features_100000.h5', classifier=classif))