# Clinical trial generation
# Author: Juan Romero


from seizure_diary_generation import generate_NV_model_patient_pop_params
from seizure_diary_generation import generate_baseline_seizure_diaries
from seizure_diary_generation import generate_maintenance_seizure_diaries
from seizure_diary_generation import apply_percent_effects_to_seizure_diaries
from endpoint_functions import calculate_MPC_p_value
import numpy as np


def generate_one_trial_seizure_diaries(num_placebo_arm_patients,
                                       num_treatment_arm_patients,
                                       num_baseline_months,
                                       num_maintenance_months,
                                       baseline_time_scale,
                                       maintenance_time_scale,
                                       minimum_cumulative_baseline_seizure_count,
                                       placebo_percent_effect_mean,
                                       placebo_percent_effect_std_dev,
                                       drug_efficacy_presence,
                                       drug_percent_effect_mean=None,
                                       drug_percent_effect_std_dev=None):
    
    '''

    This function generates four separate 2D Numpy arrays representing seizure diaries from 
    the baseline and maintenance periods  over the placebo and treatement arms of a clinical trial.

    Inputs:

        1) num_placebo_arm_patients:

            (int) - the number of patients in the placebo arm of a clinical trial

        2) num_treatment_arm_patients

            (int) - the number of patients in the treatment arm of a clinical trial

        3) num_baseline_months

            (int) - the number of months in the baseline period of a clinical trial

        4) num_maintenance_months

            (int) - the number of months in the maintenance period of a clinical trial

        5) baseline_time_scale

            (string) - the time scale on which the seizure counts in each seizure diary are generated
                       in the baseline period. This can currently be one of two values: 'daily' or 'weekly'.
                       Any other value will cause a ValueError exception.

        6) maintenance_time_scale

            (string) - the time scale on which the seizure counts in each seizure diary are generated
                       in the maintenance period. This can currently be one of two values: 'daily' or 'weekly'.
                       Any other value will cause a ValueError exception.

        7) minimum_cumulative_baseline_seizure_count

            (int) - the minimum number of seizures every individual patient will have in the baseline period

        8) placebo_percent_effect_mean

            (float) - the mean of the placebo percent effect which is gaussian-distributed over each patient

        9) placebo_percent_effect_std_dev

            (float) - the standard deviation of the placebo percent effect which is gaussian-distributed over
                      each patient

        10) drug_efficacy_presence

            (boolean) - a boolean that decides whether or not the drug effect is applied to the maintenance-period
                        seizure diaries in the treatment arm

        11) drug_percent_effect_mean

            (float, OPTIONAL) - the mean of the drug percent effect which is gaussian-distributed over each patient

        12) drug_percent_effect_std_dev

            (float, OPTIONAL) - the standard deviation of the drug percent effect which is gaussian-distributed over
                                each patient

    Outputs:

        1) placebo_arm_baseline_seizure_diaries:

            (2D Numpy array) - seizure diaries generated for the baseline period of the placebo arm of a clinical trial
        
        2) placebo_arm_maintenance_seizure_diaries:

            (2D Numpy array) - seizure diaries generated for the maintenance period of the placebo arm of a clinical trial
    
        3) treatment_arm_baseline_seizure_diaries:

            (2D Numpy array) - seizure diaries generated for the baseline period of the treatment arm of a clinical trial
        
        4) treatment_arm_maintenance_seizure_diaries:

            (2D Numpy array) - seizure diaries generated for the maintenance period of the treatment arm of a clinical trial

    '''

    # [placebo_arm_NV_model_monthly_means,
    #  placebo_arm_NV_model_monthly_std_devs] = \
    #      generate_NV_model_patient_pop_params(num_placebo_arm_patients)
    
    # [treatment_arm_NV_model_monthly_means,
    #  treatment_arm_NV_model_monthly_std_devs] = \
    #      generate_NV_model_patient_pop_params(num_treatment_arm_patients)

    [placebo_arm_NV_model_monthly_means,
     placebo_arm_NV_model_monthly_std_devs] = \
         generate_NV_model_patient_pop_params(num_placebo_arm_patients)
    
    [treatment_arm_NV_model_monthly_means,
     treatment_arm_NV_model_monthly_std_devs] = \
         generate_NV_model_patient_pop_params(num_treatment_arm_patients)
    
    placebo_arm_baseline_seizure_diaries = \
        generate_baseline_seizure_diaries(placebo_arm_NV_model_monthly_means,
                                          placebo_arm_NV_model_monthly_std_devs,
                                          baseline_time_scale,
                                          num_baseline_months,
                                          minimum_cumulative_baseline_seizure_count)
    
    treatment_arm_baseline_seizure_diaries = \
        generate_baseline_seizure_diaries(treatment_arm_NV_model_monthly_means,
                                          treatment_arm_NV_model_monthly_std_devs,
                                          baseline_time_scale,
                                          num_baseline_months,
                                          minimum_cumulative_baseline_seizure_count)

    placebo_arm_maintenance_seizure_diaries = \
        generate_maintenance_seizure_diaries(placebo_arm_NV_model_monthly_means,
                                             placebo_arm_NV_model_monthly_std_devs,
                                             maintenance_time_scale,
                                             num_maintenance_months)
    
    treatment_arm_maintenance_seizure_diaries = \
        generate_maintenance_seizure_diaries(treatment_arm_NV_model_monthly_means,
                                             treatment_arm_NV_model_monthly_std_devs,
                                             maintenance_time_scale,
                                             num_maintenance_months)

    placebo_percent_effects = \
        np.random.normal(placebo_percent_effect_mean,
                         placebo_percent_effect_std_dev,
                         num_placebo_arm_patients)

    placebo_arm_maintenance_seizure_diaries = \
        apply_percent_effects_to_seizure_diaries(placebo_arm_maintenance_seizure_diaries,
                                                 placebo_percent_effects)
    
    treatment_arm_maintenance_seizure_diaries = \
        apply_percent_effects_to_seizure_diaries(treatment_arm_maintenance_seizure_diaries,
                                                 placebo_percent_effects)
    
    if(drug_efficacy_presence == True):

        drug_percent_effects = \
            np.random.normal(drug_percent_effect_mean,
                             drug_percent_effect_std_dev,
                             num_treatment_arm_patients)
        
        treatment_arm_maintenance_seizure_diaries = \
            apply_percent_effects_to_seizure_diaries(treatment_arm_maintenance_seizure_diaries,
                                                     drug_percent_effects)
    
    return [placebo_arm_baseline_seizure_diaries,
            placebo_arm_maintenance_seizure_diaries,
            treatment_arm_baseline_seizure_diaries,
            treatment_arm_maintenance_seizure_diaries]


if(__name__=='__main__'):

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

    drug_efficacy_presence = True

    [placebo_arm_baseline_seizure_diaries,
     placebo_arm_maintenance_seizure_diaries,
     treatment_arm_baseline_seizure_diaries,
     treatment_arm_maintenance_seizure_diaries] = \
         generate_one_trial_seizure_diaries(num_placebo_arm_patients,
                                            num_drug_arm_patients,
                                            num_baseline_months,
                                            num_maintenance_months,
                                            baseline_time_scale,
                                            maintenance_time_scale,
                                            minimum_cumulative_baseline_seizure_count,
                                            placebo_percent_effect_mean,
                                            placebo_percent_effect_std_dev,
                                            drug_efficacy_presence,
                                            drug_percent_effect_mean,
                                            drug_percent_effect_std_dev)

    MPC_p_value = \
            calculate_MPC_p_value(baseline_time_scale,
                                  maintenance_time_scale,
                                  placebo_arm_baseline_seizure_diaries,
                                  placebo_arm_maintenance_seizure_diaries,
                                  treatment_arm_baseline_seizure_diaries,
                                  treatment_arm_maintenance_seizure_diaries)
    
    print(100*MPC_p_value)