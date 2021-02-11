from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from agentMET4FOF.agentMET4FOF.streams import DataStreamMET4FOF
import h5py
import numpy as np
import pandas as pd
import requests
import os
import sys

from agentMET4FOF_ml_extension.util.calc_outlier import get_num_outliers_df


class IRIS_Datastream(DataStreamMET4FOF):
    def __init__(self):
        iris = datasets.load_iris()
        self.set_data_source(quantities=iris.data, target=iris.target)

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
        data= f['Sensor_Data']
        data= data[:,:,:data.shape[2]-1] #drop last cycle
        data_inputs_np = np.zeros([data.shape[2],data.shape[1],data.shape[0]])
        for i in range(data.shape[0]):
            sensor_dt = data[i].transpose()
            data_inputs_np[:,:,i] = sensor_dt

        #prepare target var
        target=list(np.zeros(data_inputs_np.shape[0]))          # Making the target list which takes into account number of cycles, which-
        for i in range(data_inputs_np.shape[0]):                # goes from 0 to 100, and has number of elements same as number of cycles.
            target[i]=(i/(data_inputs_np.shape[0]-1))*100

        target_matrix = pd.DataFrame(target)        # Transforming list "target" into data frame "target matrix"
        data_inputs_np = self.convert_SI(data_inputs_np)
        self.set_data_source(quantities=data_inputs_np, target=target_matrix)

    def convert_SI(self, sensor_ADC):
        sensor_SI = sensor_ADC
        for i in range(sensor_ADC.shape[2]):
            sensor_SI[:,:,i]=((sensor_ADC[:,:,i]*self.gain[i])+self.offset[i])*self.b[i]*self.k[i]
        return sensor_SI

class Liveline_DataStream(DataStreamMET4FOF):
    def __init__(self, dataset_folder = "multi-stage-dataset/",
                 output_stage=1,
                 input_stage=1,
                 upper_quartile=80, train_size=0.5,
                 apply_scaling=True):

        lower_quartile = 100 - upper_quartile
        df = pd.read_csv(dataset_folder+"continuous_factory_process.csv")

        #drop columns
        df = df.drop(["time_stamp"], axis=1)
        column_names = df.columns

        # filter Y columns
        if output_stage == 1:
            Y_df_actual = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Actual" in name)]]
            Y_df_setpoint = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Setpoint" in name)]]
            Y_df_actual.columns =[str(i) for i in range(len(Y_df_actual.columns))]
            Y_df_setpoint.columns = [str(i) for i in range(len(Y_df_setpoint.columns))]
            Y_df = (Y_df_setpoint-Y_df_actual).abs()

            # Y_df = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Actual" in name)]]

        else:
            Y_df_actual = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Actual" in name)]]
            Y_df_setpoint = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Setpoint" in name)]]
            Y_df_actual.columns =[str(i) for i in range(len(Y_df_actual.columns))]
            Y_df_setpoint.columns = [str(i) for i in range(len(Y_df_setpoint.columns))]
            Y_df = (Y_df_setpoint-Y_df_actual).abs()

            # Y_df = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Actual" in name)]]
        # self.Y_df = Y_df
        # filter X columns
        if input_stage ==1:
            X_df = df[[col_name for col_name in df.columns if ("Machine1" in col_name) or ("Machine2" in col_name) or ("Machine3" in col_name)]]
        elif input_stage ==2:
            X_df = df[[col_name for col_name in df.columns if ("Machine4" in col_name) or ("Machine5" in col_name)]]
        else:
            X_df = df[[col_name for col_name in df.columns if ("Machine1" in col_name) or ("Machine2" in col_name) or ("Machine3" in col_name) or ("Machine4" in col_name) or ("Machine5" in col_name)]]

        X_columns = X_df.columns
        X = X_df.values
        Y = Y_df.values
        self.Y_df_vals = Y.copy()

        num_examples = Y.shape[0]
        self.num_examples = num_examples

        total_y_dims = Y.shape[-1]

        self.y_levels = get_num_outliers_df(self.Y_df_vals)


        unhealthy_index = (np.argwhere(self.y_levels>=1)).flatten()
        healthy_index = (np.argwhere(self.y_levels==0)).flatten()
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

        self.set_data_source(quantities={"train":x_train,"test":x_test,"ood":x_ood},
                             target={"train":y_train,"test":y_test,"ood":y_ood})

        self.X_columns = X_columns
