# Seizure diary generation
# Author: Juan Romero


import numpy as np
import sys


def generate_seizure_diary(mu, 
                           sigma, 
                           num_counts, 
                           time_scaling_product):

    var = np.square(sigma)

    if(var <= mu):

        print([mu, var])
        raise ValueError('The patient with [mu, sigma]: [' + str(mu) + ', ' + str(sigma) + '] is not overdispersed')

    mu_squared = np.square(mu)
    overdispersion = (var - mu)/mu_squared

    n = 1/overdispersion
    odds = overdispersion*mu

    if(time_scaling_product != 1):
        seizure_diary = np.random.poisson(np.random.gamma(n*time_scaling_product, odds, num_counts))
    else:
        seizure_diary = np.random.poisson(np.random.gamma(n, odds, num_counts))

    return seizure_diary


def generate_seizure_diary_with_minimum_cumulative_count(mu, 
                                                         sigma, 
                                                         num_counts,
                                                         time_scaling_product,
                                                         minimum_cumulative_seizure_count):

    acceptable_cumulative_count = False

    while(not acceptable_cumulative_count):

        seizure_diary = \
            generate_seizure_diary(mu, 
                                   sigma, 
                                   num_counts, 
                                   time_scaling_product)

        cumulative_seizure_count = np.sum(seizure_diary)

        if(cumulative_seizure_count >= minimum_cumulative_seizure_count):
            acceptable_cumulative_count = True

    return seizure_diary


def recalculate_seizure_diary_time_scales(time_axis_resolution_change_direction,
                                          num_finer_resolution_counts_within_one_coarser_resolution_count,
                                          num_original_time_scale_counts = None):

    num_new_time_scale_counts = None

    if(time_axis_resolution_change_direction == 'zoom in'):

        time_scaling_product = 1/num_finer_resolution_counts_within_one_coarser_resolution_count

        if(num_original_time_scale_counts != None):

            num_new_time_scale_counts = num_original_time_scale_counts*num_finer_resolution_counts_within_one_coarser_resolution_count
    
    elif(time_axis_resolution_change_direction == 'zoom out'):

        time_scaling_product = num_finer_resolution_counts_within_one_coarser_resolution_count

        if(num_original_time_scale_counts  != None):

            scaling_is_possible = num_original_time_scale_counts % num_finer_resolution_counts_within_one_coarser_resolution_count == 0

            if(scaling_is_possible):

                num_new_time_scale_counts = int(num_original_time_scale_counts/num_finer_resolution_counts_within_one_coarser_resolution_count)
            
            else:

                raise ValueError('Cannot zoom out on time axis due to incompatiblity between the new time scale and the number of original time scale counts')

    elif(time_axis_resolution_change_direction == 'no change'):

        num_new_time_scale_counts = num_original_time_scale_counts
        time_scaling_product = 1

    else:
        
        raise ValueError('\'' + time_axis_resolution_change_direction + '\' is not an acceptable value for the \'time_axis_resolution_change_direction\' parameter in \'recalculate_seizure_diary_time_scales()\'')

    return [time_scaling_product, num_new_time_scale_counts]


def choose_time_scale_given_monthly_parameters(seizure_diary_time_scale,
                                               num_months_in_seizure_diary):

    num_weeks_within_a_month = 4
    num_days_within_a_month  = 28

    if(seizure_diary_time_scale == 'daily'):

        num_finer_resolution_counts_within_one_coarser_resolution_count = num_days_within_a_month
    
    elif(seizure_diary_time_scale == 'weekly'):

        num_finer_resolution_counts_within_one_coarser_resolution_count = num_weeks_within_a_month
    
    else:

        raise ValueError('The \'seizure_diary_time_scale\' parameter ' + seizure_diary_time_scale + ' is not included within the code.')

    [time_scaling_product, 
     num_new_time_scale_counts] = \
         recalculate_seizure_diary_time_scales('zoom in',
                                               num_finer_resolution_counts_within_one_coarser_resolution_count,
                                               num_months_in_seizure_diary)
    
    return [time_scaling_product, 
            num_new_time_scale_counts]


def generate_baseline_seizure_diaries(monthly_mu_array,
                                      monthly_sigma_array,
                                      baseline_time_scale,
                                      num_baseline_months,
                                      minimum_cumulative_baseline_seizure_count):

    num_patients = len(monthly_mu_array)

    if(num_patients != len(monthly_sigma_array)):

        raise ValueError('The \'monthly_mu_array\' and \'monthly_sigma_array\' parameters must be the same length.')

    [baseline_time_scaling_product, 
     num_new_time_scale_baseline_counts] = \
         choose_time_scale_given_monthly_parameters(baseline_time_scale,
                                                    num_baseline_months)
    
    baseline_seizure_diaries = np.zeros((num_patients, num_new_time_scale_baseline_counts), dtype=int)

    for patient_index in range(num_patients):
        baseline_seizure_diaries[patient_index, :] = \
            generate_seizure_diary_with_minimum_cumulative_count(monthly_mu_array[patient_index], 
                                                                 monthly_sigma_array[patient_index], 
                                                                 num_new_time_scale_baseline_counts,
                                                                 baseline_time_scaling_product,
                                                                 minimum_cumulative_baseline_seizure_count)

    return baseline_seizure_diaries


def generate_maintenance_seizure_diaries(monthly_mu_array,
                                         monthly_sigma_array,
                                         maintenance_time_scale,
                                         num_maintenance_months):

    num_patients = len(monthly_mu_array)

    if(num_patients != len(monthly_sigma_array)):

        raise ValueError('The \'monthly_mu_array\' and \'monthly_sigma_array\' parameters must be the same length.')

    [maintenance_time_scaling_product, 
     num_new_time_scale_maintenance_counts] = \
         choose_time_scale_given_monthly_parameters(maintenance_time_scale,
                                                    num_maintenance_months)
    
    maintenance_seizure_diaries = np.zeros((num_patients, num_new_time_scale_maintenance_counts), dtype=int)

    for patient_index in range(num_patients):

        maintenance_seizure_diaries[patient_index, :] = \
            generate_seizure_diary(monthly_mu_array[patient_index], 
                                   monthly_sigma_array[patient_index], 
                                   num_new_time_scale_maintenance_counts, 
                                   maintenance_time_scaling_product)
    
    return maintenance_seizure_diaries


def generate_NV_model_patient_pop_params(num_patients):

    shape = 111.313
    scale = 296.728
    alpha = 296.339
    beta  = 243.719

    daily_to_monthly_conversion_factor = 28

    NV_model_monthly_means    = np.zeros(num_patients)
    NV_model_monthly_std_devs = np.zeros(num_patients)

    for patient_index in range(num_patients):

        daily_n = np.random.gamma(shape, 1/scale)
        daily_p = np.random.beta(alpha, beta)

        odds = (1 - daily_p)/daily_p

        daily_mean    = daily_n*odds
        daily_std_dev = np.sqrt(daily_mean/daily_p)

        monthly_mean    =          daily_to_monthly_conversion_factor*daily_mean
        monthly_std_dev = np.sqrt(daily_to_monthly_conversion_factor)*daily_std_dev

        if(np.square(monthly_std_dev) == monthly_mean):
            print(daily_n, daily_p)

        NV_model_monthly_means[patient_index]    = monthly_mean
        NV_model_monthly_std_devs[patient_index] = monthly_std_dev
    
    return [NV_model_monthly_means, NV_model_monthly_std_devs]


def apply_percent_effects_to_seizure_diaries(seizure_diaries, percent_effects):

    [num_patients, num_counts_per_diary] = seizure_diaries.shape
    new_seizure_diaries = np.zeros((num_patients, num_counts_per_diary), dtype=int)

    for patient_index in range(num_patients):

        percent_effect = percent_effects[patient_index]

        for count_index in range(num_counts_per_diary):

            old_count = seizure_diaries[patient_index, count_index]
            
            new_seizure_diaries[patient_index, count_index] = \
                old_count - np.sign(old_count)*np.sum(np.random.uniform(0, 1, old_count) < np.abs(percent_effect))
    
    return new_seizure_diaries

