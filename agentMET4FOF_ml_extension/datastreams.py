import io
import os
import pickle
import sys
import zipfile

import h5py
import numpy as np
import pandas as pd
import requests
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .util.tukey_fence import tukey_fence_outlier

from ..agentMET4FOF.agentMET4FOF.streams import DataStreamMET4FOF
from .util.calc_outlier import get_num_outliers_df


class IRIS_Datastream(DataStreamMET4FOF):
    def __init__(self):
        iris = datasets.load_iris()
        self.set_data_source(quantities=iris.data, target=iris.target)


class BOSTON_Datastream(DataStreamMET4FOF):
    def __init__(self):
        boston = datasets.load_boston()
        self.set_data_source(quantities=boston.data, target=boston.target)


class STRATH_Datastream(DataStreamMET4FOF):
    def __init__(self, phase="heating"):

        afrc_data_url = 'https://zenodo.org/record/3405265/files/STRATH%20radial%20forge%20dataset%20v2.zip?download=1'

        scopes_data_path = 'Data/STRATH radial forge dataset 11Sep19'  # folder for dataset

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

        # load each part's data as a dataframe to a list
        for filename in os.listdir(scopes_data_path):
            if "Scope" in filename and "csv" in filename:
                file_csv = pd.read_csv(scopes_data_path + '/' + filename, encoding='cp1252')
                data_inputs_list.append(file_csv)

        input_pd = pd.concat(data_inputs_list, axis=0, ignore_index=True)
        input_col = input_pd.columns
        # ================target dimensions====================
        output_pd = pd.read_excel(scopes_data_path + "/" + "CMMData.xlsx")

        nominal_pd = output_pd.iloc[0, 4:]
        actual_pd = output_pd.iloc[3:, 4:]
        error_pd = (actual_pd - nominal_pd)
        target_col = actual_pd.columns

        # outliers = tukey_fence_outlier(np.abs(error_pd.values))
        outliers = get_num_outliers_df(np.abs(error_pd.values))

        arg_outliers = np.argwhere(outliers >= 2)[:, 0]
        arg_inliers = np.argwhere(outliers == 0)[:, 0]

        # ===================segment heat/forge phases=======================
        # define sensors for heat/forge phases
        heating_sensors_columns = ['IP_ActSpd [mm/min]',
                                   'IP_ActPos [mm]',
                                   # 'ForgingBox_Temp',
                                   # 'TMP_Ind_U1 [°C]',
                                   # 'TMP_Ind_F [°C]'
                                   ]
        # print()
        forging_sensors_columns = [
            # 'Power [kW]',
            # 'Force [kN]',
            # 'Schlagzahl [1/min]',
            # 'EXZ_pos [deg]',
            # 'A_ACT_Force [kN]',
            # 'A_ACTpos [mm]', # could be relevant
            # 'A_ACTspd [mm/min]',
            # 'DB_ACTpos [mm]', # could be relevant
            # 'DB_ACTspd [mm/min]',
            # 'DB_ACT_Force [kN]',
            # 'L_ACTpos [mm]', # could be relevant
            # 'L_ACTspd [mm/min]',
            # 'R_ACTpos [mm]',
            # 'R_ACTspd [mm/min]',
            # 'SBA_ActPos [mm]',
            # 'W2 Durchfluss [l]', # fft this seems relevant?
            'W1 Durchfluss [l]',
            # 'Feedback L [%]',
            # 'Feedback SBA [%]',
            # 'Feedback R [%]',
            # 'Feedback DB [%]'
        ]
        # forging_sensors_columns = ['Schlagzahl [1/min]']
        heating_segment_signal = input_pd['$U_GH_HEATON_1 (U25S0)']  # digital signal for heating

        # prepare heating segmentation
        heating_start = np.where(heating_segment_signal.diff() == 1)[0][:-1]
        heating_stop = np.where(heating_segment_signal.diff() == -1)[0]
        duration_heat = heating_stop - heating_start
        min_duration_heat = np.min(duration_heat)  # truncate to shortest heat duration
        heating_sensors = [input_pd[heating_sensors_columns][on:off].iloc[:min_duration_heat] for on, off in
                           zip(heating_start, heating_stop)]

        # prepare forging segmentation
        forging_segment_signal = input_pd["Force [kN]"].values
        trigger_val = 0.3
        forging_start = np.flatnonzero(
            (forging_segment_signal[:-1] < trigger_val) & (forging_segment_signal[1:] > trigger_val)) + 1
        forging_stop = np.flatnonzero(
            (forging_segment_signal[:-1] > trigger_val) & (forging_segment_signal[1:] < trigger_val)) + 1
        duration_forge = forging_stop - forging_start
        min_duration_forge = np.min(duration_forge)  # truncate to shortest forge duration
        forging_sensors = [input_pd[forging_sensors_columns][on:off].iloc[:min_duration_forge] for on, off in
                           zip(forging_start, forging_stop)]

        segmentation_points = pd.DataFrame({"heating_start": heating_start, "heating_stop": heating_stop,
                                            "forging_start": forging_start, "forging_stop": forging_stop})

        heating_sensors_np = np.array(heating_sensors)
        forging_sensors_np = np.array(forging_sensors)
        self.xlabels = input_col
        self.ylabels = target_col
        self.arg_outliers = arg_outliers
        self.arg_inliers = arg_inliers

        if phase == "heating":
            self.set_data_source(quantities=heating_sensors_np, target=outliers)
        elif phase == "forging":
            self.set_data_source(quantities=forging_sensors_np, target=outliers)


class ZEMA_DataStream(DataStreamMET4FOF):
    url = ""
    path = ""
    _urls_choices = {"3": "https://zenodo.org/record/2702226/files/axis3_2kHz.h5",
                     "5": "https://zenodo.org/record/2702226/files/axis5_2kHz.h5",
                     "7": "https://zenodo.org/record/2702226/files/axis7_2kHz.h5"
                     }

    def get_filename(self):
        return os.path.join(self.path, self.url.split('/')[-1])

    def do_download(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        with open(self.get_filename(), "wb") as f:
            response = requests.get(self.url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write(
                        "\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()

    def __init__(self, axis=3):

        self.url = self._urls_choices[str(axis)]
        self.path = os.path.join(os.path.dirname(__file__), "dataset")

        # Check if the file is existing already, if not download the file.

        if os.path.isfile(self.get_filename()):
            print("Data already exist.\n")
        else:
            print("Download data...")
            self.do_download()
            print("Download finished.\n")

        f = h5py.File(self.get_filename(), 'r')

        # Order of sensors in the picture is different from the order in
        # imported data, which will be followed.
        self.offset = [0, 0, 0, 0, 0.00488591, 0.00488591, 0.00488591,
                       0.00488591, 1.36e-2, 1.5e-2, 1.09e-2]
        self.gain = [5.36e-9, 5.36e-9, 5.36e-9, 5.36e-9, 3.29e-4, 3.29e-4,
                     3.29e-4, 3.29e-4, 8.76e-5, 8.68e-5, 8.65e-5]
        self.b = [1, 1, 1, 1, 1, 1, 1, 1, 5.299641744, 5.299641744, 5.299641744]
        self.k = [250, 1, 10, 10, 1.25, 1, 30, 0.5, 2, 2, 2]
        self.units = ['[Pa]', '[g]', '[g]', '[g]', '[kN]', '[bar]', '[mm/s]',
                      '[A]', '[A]', '[A]', '[A]']
        self.labels = ['Microphone', 'Vibration plain bearing',
                       'Vibration piston rod', 'Vibration ball bearing',
                       'Axial force', 'Pressure', 'Velocity', 'Active current',
                       'Motor current phase 1', 'Motor current phase 2',
                       'Motor current phase 3']

        # prepare sensor data
        list(f.keys())
        data = f['Sensor_Data']
        data = data[:, :, :data.shape[2] - 1]  # drop last cycle
        data_inputs_np = np.zeros([data.shape[2], data.shape[1], data.shape[0]])
        for i in range(data.shape[0]):
            sensor_dt = data[i].transpose()
            data_inputs_np[:, :, i] = sensor_dt

        # prepare target var
        target = list(np.zeros(
            data_inputs_np.shape[0]))  # Making the target list which takes into account number of cycles, which-
        for i in range(
                data_inputs_np.shape[0]):  # goes from 0 to 1, and has number of elements same as number of cycles.
            target[i] = (i / (data_inputs_np.shape[0] - 1))

        target_matrix = pd.DataFrame(target)  # Transforming list "target" into data frame "target matrix"
        data_inputs_np = self.convert_SI(data_inputs_np)
        self.set_data_source(quantities=data_inputs_np, target=target_matrix)

    def convert_SI(self, sensor_ADC):
        sensor_SI = sensor_ADC
        for i in range(sensor_ADC.shape[2]):
            sensor_SI[:, :, i] = ((sensor_ADC[:, :, i] * self.gain[i]) + self.offset[i]) * self.b[i] * self.k[i]
        return sensor_SI

class PRONOSTIA_DataStream(DataStreamMET4FOF):
    url = ""
    path = ""

    def __init__(self, base_folder="FEMTOBearingDataSet/", pickle_folder="pickles/",
                 select_sensors=[], axis="1_1", sensor_type="acc", overwrite=False, **kwargs):

        super(PRONOSTIA_DataStream, self).__init__(**kwargs)

        # init folder paths
        self.bearing_name = "Bearing" + axis + "_" + sensor_type
        self.base_folder = base_folder
        self.training_folder = self.base_folder + "Training_set/Learning_set/"
        self.test_folder = self.base_folder + "Validation_set/Full_Test_Set/"
        self.pickle_folder = self.base_folder + pickle_folder
        self.training_bearings = [self.training_folder + bearing for bearing in os.listdir(self.training_folder)]
        self.test_bearings = [self.test_folder + bearing for bearing in os.listdir(self.test_folder)]

        # check if exists
        self.prepare_pickle(overwrite)

        # download files
        # if asset_index == 0:
        #     url = "https://zenodo.org/record/2702226/files/axis3_2kHz.h5"
        # elif asset_index == 1:
        #     url = "https://zenodo.org/record/2702226/files/axis5_2kHz.h5"
        # elif asset_index == 2:
        #     url = "https://zenodo.org/record/2702226/files/axis7_2kHz.h5"
        #
        # self.url = url
        # self.path = os.path.join(os.path.dirname(__file__), "dataset")

        # Check if the file is existing already, if not download the file.

        # if os.path.isfile(self.get_filename()):
        #     print("Data already exist.\n")
        # else:
        #     print("Download data...")
        #     self.do_download()
        #     print("Download finished.\n")

        # Order of sensors in the picture is different from the order in
        # imported data, which will be followed.

        self.units = ['[]', '[]']

        self.labels = ['Horizontal Vibration',"Vertical Vibration"]

        # prepare sensor data
        if sensor_type == "temp":
            data_inputs_np = self.load_pickle()[:, :, -1]
            data_inputs_np = np.expand_dims(data_inputs_np, axis=2)
        else:
            data_inputs_np = self.load_pickle()[:, :, -2:]

        target_RUL = self.infer_true_RUL(data_inputs_np.shape[0])
        target_RUL = pd.DataFrame(target_RUL)

        # select certain sensors only
        if len(select_sensors) == 0:
            self.select_sensors = np.arange(data_inputs_np.shape[-1])
        else:
            self.select_sensors = select_sensors

        # apply filter to selected sensors only
        self.set_data_source(quantities=data_inputs_np[:, :, self.select_sensors], target=target_RUL)

        self.units = np.array(self.units)[self.select_sensors]
        self.labels = np.array(self.labels)[self.select_sensors]

    def print_data_shapes(self):
        all_pickles = os.listdir(self.pickle_folder)
        for pickle_file in all_pickles:
            bearing_np = pickle.load(open(self.pickle_folder + pickle_file, 'rb'))
            print(pickle_file + " " + str(bearing_np.shape))

    def prepare_pickle(self, overwrite=False):
        # base_folder = "FEMTOBearingDataSet/"
        training_folder = self.base_folder + "Training_set/Learning_set/"
        test_folder = self.base_folder + "Validation_set/Full_Test_Set/"
        pickle_folder = self.base_folder + "pickles/"
        training_bearings = [training_folder + bearing for bearing in os.listdir(training_folder)]
        test_bearings = [test_folder + bearing for bearing in os.listdir(test_folder)]

        # create new folders if not existing
        for folder in [self.base_folder, training_folder, test_folder, pickle_folder]:
            if not os.path.exists(folder):
                os.mkdir(folder)

        # dynamically determine the seperator
        def determine_sep(filepath):
            temp_cycle_file = pd.read_csv(filepath, sep=None, iterator=True)
            return temp_cycle_file._engine.data.dialect.delimiter

        # read csv files
        for bearing_folder in tqdm(training_bearings + test_bearings):
            bearing_name = bearing_folder.split('/')[-1]
            csv_files = os.listdir(bearing_folder)

            # filter file based on sensor type (acc or temp), and separator is provided for each sensor type.
            for sensor_type in ["acc", "temp"]:

                # check whether to parse based on existent file, or parse anyways based on `overwrite`
                output_file = bearing_name + "_" + sensor_type + ".p"
                if not os.path.exists(pickle_folder + output_file) or overwrite:
                    bearing_np = []
                    for csv_id, csv_file in tqdm(
                            enumerate([csv_file for csv_file in csv_files if sensor_type in csv_file])):
                        full_csv_path = bearing_folder + "/" + csv_file
                        if csv_id == 0:
                            sep = determine_sep(full_csv_path)

                        cycle_file = pd.read_csv(full_csv_path, sep=sep, header=None).values

                        # expel irregular shape of temperature cycle
                        if sensor_type == "temp" and cycle_file.shape != (600, 5):
                            continue

                        # append cycle to list
                        bearing_np.append(cycle_file)

                    # convert to numpy array
                    if len(bearing_np) > 0:
                        bearing_np = np.array(bearing_np)
                        pickle.dump(bearing_np, open(pickle_folder + output_file, 'wb'))

    def load_pickle(self):
        return pickle.load(open(self.pickle_folder + self.bearing_name + ".p", 'rb'))

    def infer_true_RUL(self, num_cycles):
        # prepare target var
        target = np.zeros(num_cycles)  # Making the target list which takes into account number of cycles, which-
        for i in range(num_cycles):  # goes from 0 to 100, and has number of elements same as number of cycles.
            target[i] = (i / (num_cycles - 1))
        return target.reshape(-1, 1)


class Liveline_DataStream(DataStreamMET4FOF):
    def __init__(self, dataset_folder="multi-stage-dataset/",
                 output_stage=1,
                 input_stage=1,
                 upper_quartile=80, train_size=0.5,
                 apply_scaling=True):

        lower_quartile = 100 - upper_quartile
        df = pd.read_csv(dataset_folder + "continuous_factory_process.csv")

        # drop columns
        df = df.drop(["time_stamp"], axis=1)
        column_names = df.columns

        # filter Y columns
        if output_stage == 1:
            Y_df_actual = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Actual" in name)]]
            Y_df_setpoint = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Setpoint" in name)]]
            Y_df_actual.columns = [str(i) for i in range(len(Y_df_actual.columns))]
            Y_df_setpoint.columns = [str(i) for i in range(len(Y_df_setpoint.columns))]
            Y_df = (Y_df_setpoint - Y_df_actual).abs()

            # Y_df = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Actual" in name)]]

        else:
            Y_df_actual = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Actual" in name)]]
            Y_df_setpoint = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Setpoint" in name)]]
            Y_df_actual.columns = [str(i) for i in range(len(Y_df_actual.columns))]
            Y_df_setpoint.columns = [str(i) for i in range(len(Y_df_setpoint.columns))]
            Y_df = (Y_df_setpoint - Y_df_actual).abs()

            # Y_df = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Actual" in name)]]
        # self.Y_df = Y_df
        # filter X columns
        if input_stage == 1:
            X_df = df[[col_name for col_name in df.columns if
                       ("Machine1" in col_name) or ("Machine2" in col_name) or ("Machine3" in col_name)]]
        elif input_stage == 2:
            X_df = df[[col_name for col_name in df.columns if ("Machine4" in col_name) or ("Machine5" in col_name)]]
        else:
            X_df = df[[col_name for col_name in df.columns if
                       ("Machine1" in col_name) or ("Machine2" in col_name) or ("Machine3" in col_name) or (
                                   "Machine4" in col_name) or ("Machine5" in col_name)]]

        X_columns = X_df.columns
        X = X_df.values
        Y = Y_df.values
        self.Y_df_vals = Y.copy()

        num_examples = Y.shape[0]
        self.num_examples = num_examples

        total_y_dims = Y.shape[-1]

        self.y_levels = get_num_outliers_df(self.Y_df_vals)
        y_levels = get_num_outliers_df(self.Y_df_vals)
        # outliers = tukey_fence_outlier(self.Y_df_vals)

        unhealthy_index = (np.argwhere(self.y_levels >= 1)).flatten()
        healthy_index = (np.argwhere(self.y_levels == 0)).flatten()
        Y = self.y_levels

        x_train = X[healthy_index]
        y_train = Y[healthy_index]
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=train_size)
        x_ood = X[unhealthy_index]
        y_ood = Y[unhealthy_index]

        # apply Scaling
        if apply_scaling:
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            x_ood = scaler.transform(x_ood)

            x_test = np.clip(x_test, 0, 1)
            x_ood = np.clip(x_ood, 0, 1)

        print("INPUT STAGE :" + str(input_stage))
        print("OUTPUT STAGE :" + str(output_stage))

        print("X TRAIN SHAPE: " + str(x_train.shape))
        print("X TEST SHAPE: " + str(x_test.shape))
        print("X OOD SHAPE: " + str(x_ood.shape))

        print("Y TRAIN SHAPE: " + str(y_train.shape))
        print("Y TEST SHAPE: " + str(y_test.shape))
        print("Y OOD SHAPE: " + str(y_ood.shape))

        self.set_data_source(quantities={"train": x_train, "test": x_test, "ood": x_ood},
                             target={"train": y_train, "test": y_test, "ood": y_ood})

        self.X_columns = X_columns
