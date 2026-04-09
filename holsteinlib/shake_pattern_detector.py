# Author: Oshana Dissanayake
# Date: 04/12/2023
# Description: This script contains functions for detecting shake patterns in a signal, including
#              validating rest durations, performing moving average filtering, detecting bit changes,
#              trimming zeros, finding shake indexes, calculating shake and rest durations, and 
#              preprocessing the signal.

import statistics


def validate_rest_durations(rest_arr, rest_range_lower, rest_range_upper):
    """
    Validates if all rest durations in the array fall within the specified range.

    Parameters:
    rest_arr (list): List of rest durations.
    rest_range_lower (int): Lower bound of the rest duration range.
    rest_range_upper (int): Upper bound of the rest duration range.

    Returns:
    bool: True if all rest durations are within the specified range, otherwise False.
    """
    return all(rest_range_lower <= x <= rest_range_upper for x in rest_arr)

def perform_moving_average(signal, average_out_window_len, average_out_window_threshold):
    """
    Applies a moving average filter to the signal and thresholds the result.

    Parameters:
    signal (list): The input signal.
    average_out_window_len (int): Length of the window for calculating the moving average.
    average_out_window_threshold (float): Threshold for the moving average result.

    Returns:
    list: The processed signal after applying the moving average and thresholding.
    """
    averaged_out_signal = []
    for i in range(len(signal)):
        # Determine the start and end indices for the window
        start = max(0, i - average_out_window_len // 2)
        end = min(len(signal), i + average_out_window_len // 2 + 1)
        window = signal[start:end]

        # Calculate the mean of the window
        av = statistics.mean(window)
        # Apply the threshold
        if av >= average_out_window_threshold:
            averaged_out_signal.append(1)
        else:
            averaged_out_signal.append(0)
            
    return averaged_out_signal

def detect_bit_change(window):
    """
    Detects transitions in a binary signal from 0 to 1 and from 1 to 0.

    Parameters:
    window (list): The binary signal.

    Returns:
    list: A list containing two lists - indices where transitions from 1 to 0 and from 0 to 1 occur.
    """
    zeros_to_ones = []
    ones_to_zeros = []

    # Loop through the array and check for changes
    for i in range(1, len(window)):
        if window[i] == 1 and window[i-1] == 0:
            zeros_to_ones.append(i)
        elif window[i] == 0 and window[i-1] == 1:
            ones_to_zeros.append(i)
            
    return [ones_to_zeros, zeros_to_ones]

def trim_zeros(arr):
    """
    Trims leading and trailing zeros from the array.

    Parameters:
    arr (list): The input array.

    Returns:
    list: The array with leading and trailing zeros removed.
    """
    start = 0
    end = len(arr)

    # Find the first non-zero element
    while start < end and arr[start] == 0:
        start += 1

    # Find the last non-zero element
    while end > start and arr[end-1] == 0:
        end -= 1

    return arr[start:end]

def find_first_and_last_shake_indexes(signal):
    """
    Finds the first and last indexes where a shaking pattern is detected in the signal.

    Parameters:
    signal (list): The binary signal.

    Returns:
    tuple: The first and last indexes of the shake pattern.
    """
    # find first index
    first_index = -1
    for i in range(len(signal)):
        if(i+5 >= len(signal)):
            break
        if statistics.mode(signal[i:i+5]) == 1:
            if((signal[i] == 0 and signal[i+1] == 1) or (signal[i] == 1 and signal[i+1] == 1) or (signal[i] == 1 and signal[i+1] == 0)):
                first_index = i
                break
            else:
                i += 1
                while statistics.mode(signal[i:i+5]) == 1:
                    if((signal[i] == 0 and signal[i+1] == 1) or (signal[i] == 1 and signal[i+1] == 1) or (signal[i] == 1 and signal[i+1] == 0)):
                        first_index = i
                        break
                    i += 1
            break
            
    # find last index
    last_index = -1
    for j in range(len(signal)-1, 0, -1):
        if(j-5 < 0):
            break
        if statistics.mode(signal[j-4:j+1]) == 1:
            if((signal[j] == 0 and signal[j-1] == 1) or (signal[j] == 1 and signal[j-1] == 1) or (signal[j] == 1 and signal[j-1] == 0)):
                last_index = j
                break
            else:
                j -= 1
                while statistics.mode(signal[j-4:j+1]) == 1:
                    if((signal[j] == 0 and signal[j-1] == 1) or (signal[j] == 1 and signal[j-1] == 1) or (signal[j] == 1 and signal[j-1] == 0)):
                        last_index = j
                        break
                    j -= 1
            break
    
    return first_index, last_index

def calculate_shake_rest_durations(shake_rest_indexes, len_of_cleaned_signal):
    """
    Calculates the durations of shake and rest periods based on detected transitions.

    Parameters:
    shake_rest_indexes (list): List containing two lists - indices where transitions from 1 to 0 and from 0 to 1 occur.
    len_of_cleaned_signal (int): Length of the cleaned signal.

    Returns:
    list: A list containing two lists - shake durations and rest durations.
    """
    shake_durations = []
    rest_durations = []
    
    for i in range(len(shake_rest_indexes[0])):
        rest_durations.append(shake_rest_indexes[1][i] - shake_rest_indexes[0][i])

    for i in range(-1, len(rest_durations)):
        if(i == -1):
            shake_durations.append(shake_rest_indexes[0][0])
        elif(i == (len(rest_durations) - 1)):
            shake_durations.append(len_of_cleaned_signal - shake_rest_indexes[1][i])
        else:
            shake_durations.append(shake_rest_indexes[0][i+1] - shake_rest_indexes[1][i])
            
    return [shake_durations, rest_durations]

def signal_preprocess(signal, shake_threshold, shake_count_lower, rest_count_lower, noise_cleaning_function, *noise_cleaning_function_args):
    """
    Preprocesses the signal by thresholding, trimming, and cleaning it.

    Parameters:
    signal (list): The input signal.
    shake_threshold (float): Threshold for detecting shake.
    shake_count_lower (int): Minimum number of shakes required.
    rest_count_lower (int): Minimum number of rests required.
    noise_cleaning_function (function): Function to clean noise from the signal.
    noise_cleaning_function_args: Additional arguments for the noise cleaning function.

    Returns:
    list: A list containing a status code, the cleaned signal, and the start and end indices.
    """
    signal_binary = [0 if x < shake_threshold else 1 for x in signal]

    start, end = find_first_and_last_shake_indexes(signal_binary)
    
    if(start == -1 or end == -1):
        return [0, None, None]

    # At least 1 bit should be there to represent each shake and rest period after trimming
    if len(signal_binary[start:end]) <= (shake_count_lower + rest_count_lower):
        return [0, None, None]
    else:
        # Averaging out the signal using moving average
        trimmed_signal = trim_zeros(signal_binary[start:end])
        cleaned_out_signal = noise_cleaning_function(trimmed_signal, *noise_cleaning_function_args)
        return [1, cleaned_out_signal, [start, end]]
    
def shake_pattern_detector(signal, shake_threshold=0.75, rest_count_lower=4, rest_count_upper=5, shake_count_lower=5, shake_count_upper=6, rest_range_lower=8, rest_range_upper=13, shake_range_lower=3, shake_range_upper=7, average_out_window_len=3, average_out_window_threshold=0.5):
    """
    Detects shake patterns in the signal based on specified thresholds and ranges.

    Parameters:
    signal (list): The input signal.
    shake_threshold (float): Threshold for detecting shake.
    rest_count_lower (int): Minimum number of rests required.
    rest_count_upper (int): Maximum number of rests allowed.
    shake_count_lower (int): Minimum number of shakes required.
    shake_count_upper (int): Maximum number of shakes allowed.
    rest_range_lower (int): Lower bound for rest duration.
    rest_range_upper (int): Upper bound for rest duration.
    shake_range_lower (int): Lower bound for shake duration.
    shake_range_upper (int): Upper bound for shake duration.
    average_out_window_len (int): Length of the window for calculating the moving average.
    average_out_window_threshold (float): Threshold for the moving average result.

    Returns:
    list: A list containing a status code, the start and end indices, and the combined duration of the first and last shake.
    """
    noise_cleaning_args = [average_out_window_len, average_out_window_threshold]
    
    processed_signal = signal_preprocess(signal, shake_threshold, shake_count_lower, rest_count_lower, perform_moving_average, *noise_cleaning_args)
    
    if(processed_signal[0] == 1):
        # Return shake indexes first
        shake_rest_indexes = detect_bit_change(processed_signal[1])
        
        if(len(shake_rest_indexes[1]) >= rest_count_lower):
            # Return shake durations first
            shake_durations, rest_durations = calculate_shake_rest_durations(shake_rest_indexes, len(processed_signal[1]))
            if((shake_count_lower <= len(shake_durations) <= shake_count_upper) and 
               (rest_count_lower <= len(rest_durations) <= rest_count_upper) and 
               validate_rest_durations(rest_durations, rest_range_lower, rest_range_upper)):
                win_start_index = signal.index[0]
                return [1, [win_start_index+processed_signal[2][0], win_start_index+processed_signal[2][1]], shake_durations[0] + shake_durations[-1]]
            else:
                return [0]
        else:
            return [0]
    else:
        return [0]