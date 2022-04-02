# Author: Tammie li
# Description: result record
# FilePath: \DRL\Utils\record.py


import os

def write_table_head(subject_id, model_info, dataset):
    # Create a file to save the experimental results
    file_path = os.path.join(os.getcwd(), 'ExperienceResult', f'result_S{subject_id:>02d}_{dataset}')

    with open(file_path, 'a+') as f:
        f.writelines("subject_id" + "\t\t" + 
                    "Mean" + "\t\t" + 
                    "F1" + "\t\t" +
                    "F2" + "\t\t" +
                    "D" + "\t\t" +
                    "droup_out" + "\t\t" +
                    "kernel_size" + "\t\t" +
                    "UAR" + "\t\t\t" +
                    "WAR" + "\n"
                    ) 

def write_result(subject_id, model_info, dataset, mean, uar, war):
    # write the result in table
    file_path = os.path.join(os.getcwd(), 'ExperienceResult', f'result_S{subject_id:>02d}_{dataset}')

    with open(file_path, 'a+') as f:
        f.writelines(str(subject_id) + "\t\t\t\t" + 
                    str(mean) + "\t\t\t" + 
                    str(model_info['F1']) + "\t\t" +
                    str(model_info['F2']) + "\t\t" +
                    str(model_info['D']) + "\t\t" +
                    str(model_info['droup_out']) + "\t\t\t\t" +
                    str(model_info["kernel_size"]) + "\t\t\t\t" +
                    str(uar) + "\t\t" +
                    str(war) + "\n"
                    ) 