# Endpoint functions
# Author: Juan Romero


from seizure_diary_generation import recalculate_seizure_diary_time_scales
import scipy.stats as stats
import numpy as np


def select_time_scale(seizure_diary_time_scale):

    num_weeks_within_a_month = 4
    num_days_within_a_month  = 28

    if(seizure_diary_time_scale == 'daily'):

        num_finer_resolution_counts_within_one_coarser_resolution_count = num_days_within_a_month
    
    elif(seizure_diary_time_scale == 'weekly'):

        num_finer_resolution_counts_within_one_coarser_resolution_count = num_weeks_within_a_month
    
    else:

        raise ValueError('The \'seizure_diary_time_scale\' parameter ' + seizure_diary_time_scale + ' is not included within the code.')

    [time_scaling_product, _] = \
         recalculate_seizure_diary_time_scales('zoom out',
                                               num_finer_resolution_counts_within_one_coarser_resolution_count)
    
    return time_scaling_product
    

def calculate_percent_changes(baseline_time_scale,
                              baseline_seizure_diaries,
                              maintenance_time_scale,
                              maintenance_seizure_diaries):

    '''

    This function calculates the percent change in seizure frequency (i.e., the mean of 
    the seizure counts) between the baseline and maintenance period of a patients over all
    patients. The time scale on which each seizure diary is generated is needed in case the time
    scales are different between the baseline and maintenance periods.

    Inputs:

        1) baseline_time_scale

            (string) - the time scale on which the seizure counts in each seizure diary were generated
                       in the baseline period. This can currently be one of two values: 'daily' or 'weekly'.
                       Any other value will cause a ValueError exception.
        
        2) baseline_seizure_diaries:

            (2D Numpy array) - seizure diaries generated for the baseline period of one arm of a clinical trial

        3) maintenance_time_scale

            (string) - the time scale on which the seizure counts in each seizure diary were generated
                       in the maintenance period. This can currently be one of two values: 'daily' or 'weekly'.
                       Any other value will cause a ValueError exception.
        
         2) baseline_seizure_diaries:

            (2D Numpy array) - seizure diaries generated for the maintenance period of one arm of a clinical trial

    Outputs:

        1) percent_changes

            (1D Numpy array) - the percent change in seizure frequency for each patient
                               in one arm of a clinical trial

    '''

    baseline_seizure_frequencies    = np.mean(baseline_seizure_diaries,    1)
    maintenance_seizure_frequencies = np.mean(maintenance_seizure_diaries, 1)

    percent_changes = 1 - (baseline_seizure_frequencies/maintenance_seizure_frequencies)

    if(baseline_time_scale != maintenance_time_scale):

        baseline_time_scaling_product    = select_time_scale(   baseline_time_scale)
        maintenance_time_scaling_product = select_time_scale(maintenance_time_scale)

        percent_changes = 1 - (baseline_time_scaling_product/maintenance_time_scaling_product)*(1 - percent_changes)

    return percent_changes


def calculate_MPC_p_value(baseline_time_scale,
                          maintenance_time_scale,
                          placebo_arm_baseline_seizure_diaries,
                          placebo_arm_maintenance_seizure_diaries,
                          treatment_arm_baseline_seizure_diaries,
                          treatment_arm_maintenance_seizure_diaries):

    '''

    This function calculates the p-value for a clinical trial given the seizure diaries.
    The time scales on which the seizure diaries are needed as inputs in case the seizure diaries
    are generated on different time scales (e.g., weekly baseline seizure counts and daily maintenance
    seizure counts).

    Inputs:

        1) baseline_time_scale

            (string) - the time scale on which the seizure counts in each seizure diary were generated
                       in the baseline period. This can currently be one of two values: 'daily' or 'weekly'.
                       Any other value will cause a ValueError exception.

        2) maintenance_time_scale

            (string) - the time scale on which the seizure counts in each seizure diary were generated
                       in the maintenance period. This can currently be one of two values: 'daily' or 'weekly'.
                       Any other value will cause a ValueError exception.

        3) placebo_arm_baseline_seizure_diaries:

            (2D Numpy array) - seizure diaries generated for the baseline period of the placebo arm of a clinical trial
        
        4) placebo_arm_maintenance_seizure_diaries:

            (2D Numpy array) - seizure diaries generated for the maintenance period of the placebo arm of a clinical trial
    
        5) treatment_arm_baseline_seizure_diaries:

            (2D Numpy array) - seizure diaries generated for the baseline period of the treatment arm of a clinical trial
        
        6) treatment_arm_maintenance_seizure_diaries:

            (2D Numpy array) - seizure diaries generated for the maintenance period of the treatment arm of a clinical trial

    Outputs:

        1) MPC_p_value:

            (float) - the p-value for a clinical trial given that the primary endpoint for the trial
                      is  the MPC (Median Percent Change)

    '''

    placebo_arm_percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  placebo_arm_baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  placebo_arm_maintenance_seizure_diaries)

    treatment_arm_percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  treatment_arm_baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  treatment_arm_maintenance_seizure_diaries)

    [_, MPC_p_value] = stats.ranksums(placebo_arm_percent_changes, treatment_arm_percent_changes)

    return MPC_p_value

