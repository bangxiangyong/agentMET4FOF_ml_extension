import pandas as pd
import os
import numpy as np

csv_folder = "publication/"

def read_csv_mean_sem(perf_key = "gmean-sdc",csv_folder = "publication/"):
    csv_folder = "publication/"

    files = os.listdir(csv_folder)
    filtered_files = [file for file in files if perf_key in file]

    csvs = {}
    for file in filtered_files:
        for stat in ["-mean-", "-sem-"]:
            if stat in file:
                csvs.update({stat: pd.read_csv(csv_folder + file)})

    return csvs

def write_latex_max_col(csvs:dict, perf_key="gmean-sdc",csv_folder = "publication/"):
    with open(csv_folder+perf_key+'-latex.txt', 'w') as latex_file:
        print("WRITING TO LATEX: " + perf_key)
        csvs["-mean-"] = csvs["-mean-"].reindex([2, 0, 1, 3, 4, 5, 6, 7, 10, 8, 9, 11, 12, 13, 14, 15]).reset_index(drop=True)
        csvs["-sem-"] = csvs["-sem-"].reindex([2, 0, 1, 3, 4, 5, 6, 7, 10, 8, 9, 11, 12, 13, 14, 15]).reset_index(drop=True)
        print(csvs["-mean-"])
        all_val = csvs["-mean-"].values[:,4:]
        max_col = []
        global_max = []
        for row_i, sub_val in enumerate([all_val[:8], all_val[8:]]):
            for col in range(sub_val.shape[1]):
                max_col += [[np.argmax(sub_val[:, col])+(row_i*8), col]]
            global_max += [np.max(sub_val)]
        for row_i,(row_mean, row_sem) in enumerate(zip(csvs["-mean-"].values[:,4:],csvs["-sem-"].values[:,4:])):
            new_string = ""
            if row_i == 8:
                print("--------------------")
                latex_file.write("--------------------"+'\n')
            for col_i, (col_mean, col_sem) in enumerate(zip(row_mean, row_sem)):
                print_mean = "%0.3f" % col_mean
                print_sem = "%0.3f" % col_sem
                # find for max col
                if [row_i,col_i] in max_col:

                    if col_mean in global_max:
                        new_string +="\\textBF{*"+ (print_mean+ "$\pm{}$"+ print_sem) + "}&"
                    else:
                        new_string += "\\textBF{" + (print_mean + "$\pm{}$" + print_sem) + "}&"
                else:
                    new_string += (print_mean + "$\pm{}$" + print_sem) + "&"
            latex_file.write(new_string[:-1]+'\n')
            print(new_string)

def write_latex_max_row(csvs:dict, perf_key="gmean-sdc",csv_folder = "publication/"):
    with open(csv_folder+perf_key+'-latex.txt', 'w') as latex_file:
        print("WRITING TO LATEX: " + perf_key)
        csvs["-mean-"] = csvs["-mean-"].reindex([2, 0, 1, 3, 4, 5, 6, 7, 10, 8, 9, 11, 12, 13, 14, 15]).reset_index(drop=True)
        csvs["-sem-"] = csvs["-sem-"].reindex([2, 0, 1, 3, 4, 5, 6, 7, 10, 8, 9, 11, 12, 13, 14, 15]).reset_index(drop=True)
        print(csvs["-mean-"])
        all_val = csvs["-mean-"].values[:,4:]
        max_row = []
        global_max = []
        # cuts subset for PRONOSTIA (8) and ZEMA (8) results
        for subtable_i, sub_val in enumerate([all_val[:8], all_val[8:]]):
            for row_i, row in enumerate(sub_val):
                # max_row += [[np.argmax(sub_val[:, col])+(row_i*8), col]]
                max_row += [[row_i+(subtable_i * 8),np.argmax(row)]]
            global_max += [np.max(sub_val)]
        for row_i,(row_mean, row_sem) in enumerate(zip(csvs["-mean-"].values[:,4:],csvs["-sem-"].values[:,4:])):
            new_string = ""
            if row_i == 8:
                print("--------------------")
                latex_file.write("--------------------"+'\n')
            for col_i, (col_mean, col_sem) in enumerate(zip(row_mean, row_sem)):
                print_mean = "%0.3f" % col_mean
                print_sem = "%0.3f" % col_sem
                # find for max col
                if [row_i,col_i] in max_row:

                    if col_mean in global_max:
                        new_string +="\\textBF{*"+ (print_mean+ "$\pm{}$"+ print_sem) + "}&"
                    else:
                        new_string += "\\textBF{" + (print_mean + "$\pm{}$" + print_sem) + "}&"
                else:
                    new_string += (print_mean + "$\pm{}$" + print_sem) + "&"
            latex_file.write(new_string[:-1]+'\n')
            print(new_string)


for perf_key in ["gmean-sser","mcc","gmean-sdc","pearson","hieq"]:
    csvs = read_csv_mean_sem(perf_key=perf_key,csv_folder=csv_folder)
    # write_latex_max_col(csvs, perf_key=perf_key, csv_folder=csv_folder)
    write_latex_max_row(csvs, perf_key=perf_key, csv_folder=csv_folder)

# np.argmax(rr,axis=1)

# 0.707$\pm{}$0.007

# for file in filtered_files:


