import copy
from tqdm import tqdm
import numpy as np
from baetorch.baetorch.util.data_model_manager import DataModelManager


class CoalitionalBAE():
    """
    Handle independently trained BAE for each sensor. Due to GPU and CPU memory load, we load and predict with each saved model sequentially.
    """

    def __init__(self, select_sensors=np.arange(11), train_bae=0, **model_params):
        self.select_sensors = select_sensors
        self.model_params = model_params
        self.data_manager = DataModelManager()
        temp_model_name = self.data_manager.encode("model", return_pickle=False, **model_params)
        self.model_name = temp_model_name[:5]+temp_model_name[-5:]
        self.train_bae = train_bae

        # preload models options
        self.preload = False
        self.preload_models = []

    def fit(self, x_train):
        temp_model_param = copy.deepcopy(self.model_params)
        print("FITTING")
        for sensor_i in tqdm(self.select_sensors):
            temp_model_param["data_params"]["select_sensors"] = np.array([sensor_i])
            temp_model_param["conv_architecture"][0] = 1
            bae_ensemble = self.data_manager.wrap(self.train_bae, "model",
                                             np.expand_dims(x_train[:,sensor_i],1), model_name=self.data_manager.encode(datatype="model", return_pickle=False, return_compact=True, **temp_model_param),
                                             **temp_model_param)

        self.decoder_sigma_enabled = bae_ensemble.decoder_sigma_enabled
        self.homoscedestic_mode = bae_ensemble.homoscedestic_mode

    def predict_samples(self, x_test, select_keys=[]):
        temp_model_param = copy.deepcopy(self.model_params)
        y_preds = []
        print("PREDICTING")
        for count_i, sensor_i in tqdm(enumerate(self.select_sensors)):
            temp_model_param["data_params"]["select_sensors"] = np.array([sensor_i])
            temp_model_param["conv_architecture"][0] = 1
            if self.preload:
                bae_ensemble = self.preload_models[count_i]
            else:
                bae_ensemble = self.data_manager.wrap(self.train_bae, "model",
                                                 [], model_name=self.data_manager.encode(datatype="model", return_pickle=False, return_compact=True, **temp_model_param),
                                                 **temp_model_param)
            if count_i == 0:
                y_preds = bae_ensemble.predict_samples(np.expand_dims(x_test[:,sensor_i],1), select_keys=select_keys)
            else:
                temp_preds = bae_ensemble.predict_samples(np.expand_dims(x_test[:,sensor_i],1), select_keys=select_keys)
                y_preds = np.concatenate((y_preds,temp_preds), axis=3)
        return y_preds

    def get_bae_models(self):
        temp_model_param = copy.deepcopy(self.model_params)
        models = [ ]
        for count_i, sensor_i in tqdm(enumerate(self.select_sensors)):
            temp_model_param["data_params"]["select_sensors"] = np.array([sensor_i])
            temp_model_param["conv_architecture"][0] = 1
            bae_ensemble = self.data_manager.wrap(self.train_bae, "model",
                                             [], model_name=self.data_manager.encode(datatype="model", return_pickle=False, return_compact=True, **temp_model_param),
                                             **temp_model_param)
            models.append(bae_ensemble)
        self.preload_models = models
        self.preload = True
        return models

    def set_cuda(self, cuda=True):
        self.cuda = cuda
        if self.preload:
            for i, model in enumerate(self.preload_models):
                self.preload_models[i].set_cuda(cuda)


