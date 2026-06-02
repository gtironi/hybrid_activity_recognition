from itertools import combinations
import math
import random
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


def find_optimal_calf_combinations_for_split_sampled(
    all_sbj_ids,
    num_to_select,
    data_amounts_df,
    split_ratio,
    n_samples=200_000,
    random_state=42,
    cv=1,
):
    """Sampled variant: evaluates ``n_samples`` random combinations and returns the best
    by the same mean-deviation criterion. Use when C(n,k) is too large to enumerate.
    Falls back to full enumeration when C(n,k) <= n_samples.
    """
    n_total = math.comb(len(all_sbj_ids), num_to_select)
    if n_total <= n_samples:
        return find_optimal_calf_combinations_for_split(
            all_sbj_ids, num_to_select, data_amounts_df, split_ratio, cv=cv
        )

    rng = random.Random(random_state)
    total_counts = data_amounts_df.sum().values[1:]
    ids_list = list(all_sbj_ids)

    deviations = {}
    seen = set()
    attempts = 0
    max_attempts = n_samples * 10

    while len(deviations) < n_samples and attempts < max_attempts:
        attempts += 1
        combination = tuple(sorted(rng.sample(ids_list, num_to_select), key=str))
        if combination in seen:
            continue
        seen.add(combination)

        comb_counts = data_amounts_df[data_amounts_df.subject_id.isin(combination)].sum().values[1:]
        train_counts = total_counts - comb_counts

        if np.any(train_counts == 0):
            continue

        label_ratios = comb_counts / train_counts
        mean_deviation = float(np.mean(np.abs(label_ratios - split_ratio)))
        deviations[mean_deviation] = combination

    if not deviations:
        return None if cv == 1 else []

    if cv == 1:
        return deviations[min(deviations.keys())]
    sorted_deviations = sorted(deviations.items())[:cv]
    return [combination for _, combination in sorted_deviations]


def partition_subjects_into_folds(all_sbj_ids, n_folds, data_amounts_df):
    """Partition subjects into n_folds non-overlapping groups for k-fold CV.

    Each fold gets a subset of subjects whose aggregated behaviour distribution
    is balanced across folds. Uses snake-order assignment on subjects sorted by
    Euclidean distance from the mean behaviour-proportion vector, so each fold
    receives subjects from every "tier" of behavioural diversity.

    Returns a list of n_folds lists of subject IDs (every subject appears in
    exactly one fold).
    """
    ids_list = list(all_sbj_ids)
    n = len(ids_list)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n < n_folds:
        raise ValueError(f"Not enough subjects ({n}) for {n_folds} folds")

    counts = data_amounts_df.set_index("subject_id")
    counts = counts.loc[[sid for sid in ids_list if sid in counts.index]]

    totals = counts.sum(axis=1).replace(0, 1)
    props = counts.div(totals, axis=0).fillna(0.0)
    mean_vec = props.mean(axis=0).values
    distances = np.linalg.norm(props.values - mean_vec, axis=1)

    order = np.argsort(distances)
    sorted_subjects = [props.index[i] for i in order]

    folds = [[] for _ in range(n_folds)]
    direction = 1
    current = 0
    for sid in sorted_subjects:
        folds[current].append(sid)
        next_fold = current + direction
        if next_fold >= n_folds:
            direction = -1
            next_fold = n_folds - 1
        elif next_fold < 0:
            direction = 1
            next_fold = 0
        current = next_fold

    return folds