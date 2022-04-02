# Author: Tammie li
# Description: Define two evaluation metrics (UAR and WAR)
# FilePath: \DRL\Utils\evaluate.py
import numpy as np


def calculate_WAR_for_single_subject(subject_id):
    pred = np.load(f'PredictionResult/{subject_id:>02d}_preds.npy')
    y = np.load(f'PredictionResult/{subject_id:>02d}_y.npy')
    sum_num = 0
    correct_num = 0
    for idx, label in enumerate(y):
        sum_num += 1
        if pred[idx] == (label):
            correct_num += 1
    result = round(correct_num / sum_num * 100, 2) 
    return result

def calculate_UAR_for_single_subject(subject_id):
    pred = np.load(f'PredictionResult/{subject_id:>02d}_preds.npy')
    y = np.load(f'PredictionResult/{subject_id:>02d}_y.npy')
    target_right_num = 0
    target_num = 0
    non_target_num = 0
    non_target_right_num = 0
    for idx, label in enumerate(y):
        if label == 1 or label == 2:
            target_num += 1
            if (pred[idx]) == label:
                target_right_num += 1
        elif label == 0:
            non_target_num += 1
            if (pred[idx]) == label:
                non_target_right_num += 1
        else:
            print("Illegal label")
            exit()
    result = round((target_right_num / target_num + non_target_right_num/non_target_num) * 50, 2)

    return result

def print_result_all_subject(subject_num):
    for subject_id in range(1, subject_num+1):
        UAR = calculate_UAR_for_single_subject(subject_id)
        WAR = calculate_WAR_for_single_subject(subject_id)
        print(subject_id, "\t\t", UAR, "\t\t", WAR)








