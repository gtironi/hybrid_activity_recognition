from itertools import combinations
import numpy as np

# Subject amounts for each train, validation, test splits
def calc_split_subject_amounts(total_subject_count, percentages):
    percentages_ = [percentages['train'], percentages['validation'], percentages['test']]
    initial_values = [round(total_subject_count * p / 100) for p in percentages_]
    diff = total_subject_count - sum(initial_values)
    for i in range(abs(diff)):
        initial_values[i % 3] += int(diff / abs(diff))
    return initial_values

# Generate all possible combinations of calves
def generate_sbj_sets(all_calves, num_to_select):
    calf_combinations = list(combinations(all_calves, num_to_select))
    return calf_combinations

# Return the optimal subject combination with generalization based on train:set label proportion
def find_optimal_calf_combinations_for_split(all_sbj_ids, num_to_select, data_amounts_df, split_ratio, cv=1):
    # Generate all possible combinations of subjects
    all_sbj_combinations = combinations(all_sbj_ids, num_to_select)
    
    total_counts = data_amounts_df.sum().values[1:]
    
    deviations = {}
    
    for combination in all_sbj_combinations:
        comb_counts = data_amounts_df[data_amounts_df.subject_id.isin(combination)].sum().values[1:]
        train_counts = total_counts - comb_counts
        
        # Checking if training data has data for all the classes
        if np.any(train_counts == 0):
            continue

        # Calculate the label ratios and their deviation from the split ratio
        label_ratios = comb_counts / train_counts
        mean_deviation = np.mean(np.abs(label_ratios - split_ratio))

        deviations[mean_deviation] = combination
    
    # Handle the case where no valid combination was found
    if not deviations:
        return None if cv == 1 else []

    if cv == 1:
        min_deviation = min(deviations.keys())
        return deviations[min_deviation]
    else:
        sorted_deviations = sorted(deviations.items())[:cv]
        return [combination for _, combination in sorted_deviations]