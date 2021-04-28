import io
import os
import requests
import zipfile
import functools

import numpy as np
import pandas as pd

from agentMET4FOF_ml_extension.util.tukey_fence import tukey_fence_outlier
# import matplotlib
# matplotlib.use("Agg")

afrc_data_url='https://zenodo.org/record/3405265/files/STRATH%20radial%20forge%20dataset%20v2.zip?download=1'

scopes_data_path = 'Data/STRATH radial forge dataset 11Sep19' #folder for dataset



def download_and_extract(url, destination, force=False):
    response = requests.get(url)
    zipDocument = zipfile.ZipFile(io.BytesIO(response.content))
    # Attempt to see if we are going to overwrite anything
    if not force:
        abort = False
        for file in zipDocument.filelist:
            if os.path.isfile(os.path.join(destination, file.filename)):
                print(file.filename,
                      'already exists. If you want to overwrite the file call the method with force=True')
                abort = True
        if abort:
            print('Zip file was not extracted')
            return

    zipDocument.extractall(destination)

# download_and_extract(afrc_data_url, scopes_data_path)


data_inputs_list = []

#load each part's data as a dataframe to a list
for filename in os.listdir(scopes_data_path):
    if "Scope" in filename and "csv" in filename:
        file_csv = pd.read_csv(scopes_data_path + '/' + filename, encoding='cp1252')
        data_inputs_list.append(file_csv)


input_pd = pd.concat(data_inputs_list, axis=0, ignore_index = True)
input_col = input_pd.columns
#================target dimensions====================
output_pd = pd.read_excel(scopes_data_path + "/" + "CMMData.xlsx")


nominal_pd = output_pd.iloc[0,4:]
actual_pd = output_pd.iloc[3:,4:]
error_pd = actual_pd-nominal_pd
target_col = actual_pd.columns

outliers = tukey_fence_outlier(error_pd)
arg_outliers = np.argwhere(outliers>=1)[:,0]

#===================segment heat/forge phases=======================
# define sensors for heat/forge phases
heating_sensors_columns = ['IP_ActSpd [mm/min]',
                           'IP_ActPos [mm]',
                           'ForgingBox_Temp',
                           'TMP_Ind_U1 [°C]',
                           'TMP_Ind_F [°C]']
forging_sensors_columns = ['Power [kW]', 'Force [kN]',
                           'Schlagzahl [1/min]',
                           'EXZ_pos [deg]',
                           'A_ACT_Force [kN]',
                           'A_ACTpos [mm]',
                           'A_ACTspd [mm/min]',
                           'DB_ACTpos [mm]',
                           'DB_ACTspd [mm/min]',
                           'DB_ACT_Force [kN]',
                           'L_ACTpos [mm]',
                           'L_ACTspd [mm/min]',
                           'R_ACTpos [mm]'
                           'R_ACTspd [mm/min]',
                           'SBA_ActPos [mm]'
                           ]
heating_segment_signal = input_pd['$U_GH_HEATON_1 (U25S0)'] # digital signal for heating

# prepare heating segmentation
heating_start = np.where(heating_segment_signal.diff()== 1)[0][:-1]
heating_stop = np.where(heating_segment_signal.diff() == -1)[0]
duration_heat = heating_stop-heating_start
min_duration_heat = np.min(duration_heat) #truncate to shortest heat duration
heating_sensors = [input_pd[heating_sensors_columns][on:off].iloc[:min_duration_heat] for on,off in zip(heating_start,heating_stop)]


# prepare forging segmentation
forging_segment_signal = input_pd["Force [kN]"].values
trigger_val = 0.3
forging_start = np.flatnonzero((forging_segment_signal[:-1] < trigger_val) & (forging_segment_signal[1:] > trigger_val))+1
forging_stop = np.flatnonzero((forging_segment_signal[:-1] > trigger_val) & (forging_segment_signal[1:] < trigger_val))+1
duration_forge = forging_stop-forging_start
min_duration_forge = np.min(duration_forge) #truncate to shortest forge duration
forging_sensors = [input_pd[forging_sensors_columns][on:off].iloc[:min_duration_forge] for on,off in zip(forging_start,forging_stop)]


segmentation_points = pd.DataFrame({"heating_start":heating_start,"heating_stop":heating_stop,
                                    "forging_start":forging_start,"forging_stop":forging_stop})

heating_sensors_np = np.array(heating_sensors)
forging_sensors_np = np.array(forging_sensors)

#===============resample==================




# print("HEAT:"+str(min_duration_heat))
# print("FORGE:"+str(min_duration_forge))


# # get a box plot
# plt.figure()
# plt.boxplot([error_pd.iloc[:,col] for col in range(len(target_col))])
# plt.xticks(np.arange(len(target_col)), target_col, rotation='vertical')
#
# plt.figure()
# for row in range(len(error_pd)):
#     plt.plot(error_pd.iloc[row], color="blue",alpha=0.25)
# plt.plot(error_pd.mean(0),color="orange")
# plt.xticks(np.arange(len(target_col)), target_col, rotation='vertical')


# #start plotting heating processes
# fig, (ax1, ax2) = plt.subplots(2,1, dpi = 120, figsize=(120,20),sharex=True)
#
# #heating plots
# verify_heating_sensor_name = "TMP_Ind_U1 [°C]"
# ax1.plot(input_pd[verify_heating_sensor_name])
# ax2.plot(heating_segment_signal)
#
# #set title
# ax1.set_title("Heating: "+ verify_heating_sensor_name)
# ax2.set_title("Heating: "+ '$U_GH_HEATON_1 (U25S0)')
#
# for index, row in segmentation_points.iterrows():
#     ax1.axvline(x=row["heating_start"], color='r', linestyle='--')
#     ax1.axvline(x=row["heating_stop"], color='r', linestyle='--')
#     ax2.axvline(x=row["heating_start"], color='r', linestyle='--')
#     ax2.axvline(x=row["heating_stop"], color='r', linestyle='--')
#
#
# #start plotting forging processes
# fig, (ax1, ax2) = plt.subplots(2,1, dpi = 120, figsize=(120,20),sharex=True)
#
# #heating plots
# verify_forging_sensor_name = "W1 Durchfluss [l]"
# ax1.plot(input_pd[verify_forging_sensor_name])
# ax2.plot(forging_segment_signal)
#
# #set title
# ax1.set_title("Forging: "+ verify_forging_sensor_name)
# ax2.set_title("Forging: "+ "Force [kN]")
#
# for index, row in segmentation_points.iterrows():
#     #draw vertical line
#     ax1.axvline(x=row["forging_start"], color='r', linestyle='--')
#     ax1.axvline(x=row["forging_stop"], color='r', linestyle='--')
#     ax2.axvline(x=row["forging_start"], color='r', linestyle='--')
#     ax2.axvline(x=row["forging_stop"], color='r', linestyle='--')



