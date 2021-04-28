import inspect
from pprint import pprint

import numpy as np
from captum.attr import DeepLift
from sklearn.pipeline import make_pipeline

from agentMET4FOF_ml_extension.ml_agents import ML_TransformAgent
from agentMET4FOF_ml_extension.util.attribution import get_attribution
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder, Autoencoder
from baetorch.baetorch.models.base_layer import DenseLayers, Conv1DLayers
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.minmax import MultiMinMaxScaler
from baetorch.baetorch.util.misc import save_bae_model, load_bae_model
from types import MethodType

# Convolutional BAE Agent
from baetorch.baetorch.util.seed import bae_set_seed


class CBAE_Agent(ML_TransformAgent):
    """
    Convolutional BAE Agent.

    total number of samples are actually use_dmm * dmm_samples

    """

    parameter_choices = {"first_nodes": [0, 100, 200, 300, 400, 500],
                         "second_nodes": [0, 100, 200, 300, 400, 500],
                         "latent_dim": [10, 50, 100],
                         "likelihood": ["1_gaussian", "homo_gaussian", "hetero_gaussian", "bernoulli", "cbernoulli"],
                         "learning_rate": [0.1, 0.01, 0.001],
                         "bae_samples": [1, 5, 10],
                         "weight_decay": [0, 0.1, 0.5, 1.0],
                         # "use_cuda": [True,False],
                         # "train_model":[True,False],
                         "num_epochs": [10, 50, 100, 150, 200, 250]
                         }

    def init_parameters(self, conv_architecture=[11, 5, 3],
                        dense_architecture=[100],
                        conv_kernel=[200, 100],
                        conv_stride=[10, 10],
                        latent_dim=50,
                        likelihood="1_gaussian",
                        learning_rate=0.01,
                        bae_samples=1,
                        random_state=123,
                        weight_decay=0.015,
                        use_cuda=True,
                        train_model=True,
                        send_train_model=False,
                        predict_train=True,
                        num_epochs=1,
                        use_dmm=False,
                        dmm_samples=5,
                        return_samples=False,
                        send_test_samples=False,
                        example_axis=None,
                        func_apply_y=False,
                        posthoc_class=None,
                        use_lr_finder=True,
                        **data_params):
        if example_axis is not None:
            self.example_axis = example_axis
        self.weight_decay = weight_decay
        self.func_apply_y = func_apply_y
        self.use_lr_finder = use_lr_finder
        self.send_test_samples = send_test_samples
        self.set_random_state(random_state)
        self.return_samples = return_samples
        self.send_train_model = send_train_model
        self.predict_train = predict_train
        self.model_params = {"conv_architecture": conv_architecture,
                             "dense_architecture": dense_architecture,
                             "conv_kernel": conv_kernel,
                             "conv_stride": conv_stride,
                             "latent_dim": latent_dim,
                             "likelihood": likelihood,
                             "bae_samples": bae_samples,
                             "num_epochs": num_epochs,
                             "use_dmm": use_dmm,
                             "dmm_samples": dmm_samples,
                             "weight_decay": weight_decay
                             }

        self.latent_dim = latent_dim
        self.dense_architecture = dense_architecture
        self.conv_architecture = conv_architecture
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.likelihood = likelihood
        self.learning_rate = learning_rate
        self.homoscedestic_mode_map = {"1_gaussian": "none", "homo_gaussian": "every",
                                       "hetero_gaussian": "none",
                                       "bernoulli": "none",
                                       "cbernoulli": "none"}
        self.likelihood_mode_map = {"1_gaussian": "gaussian", "homo_gaussian": "gaussian",
                                    "hetero_gaussian": "gaussian",
                                    "bernoulli": "bernoulli",
                                    "cbernoulli": "cbernoulli"}
        self.bae_samples = bae_samples
        self.use_cuda = use_cuda
        self.train_model = train_model
        self.num_epochs = num_epochs
        self.dmm_samples = dmm_samples

        self.posthoc_class = posthoc_class
        self.posthoc_enable = True if self.posthoc_class is not None else False

        # if use Data Model Manager is enabled
        self.init_dmm(use_dmm=use_dmm, disable_channels=["test"]) # we dont want to save the outputs for test channel as it may be too huge

        # if self.use_dmm and dmm_samples>0:
        #     self.bae_dmm_names = []
        #     for i in range(self.dmm_samples):
        #         self.bae_dmm_names.append(
        #             self._fit.__name__ + self.encode_current_params() + str(self.random_state + i))

        # show model parameters on init
        pprint(self.model_params)

    def update_bae_dmm_names(self):
        self.bae_dmm_names = []
        for i in range(self.dmm_samples):
            bae_dmm_name = self.dmm_code + str(self.random_state + i)
            self.bae_dmm_names.append(bae_dmm_name)

    def fit(self, message_data):
        """
        Checks whether DataModelManager is used.
        """
        x_train = message_data["quantities"].copy()

        # fit BAEs
        if not self.use_dmm:
            bae_model = self._fit(x_train, random_state=self.random_state)

        else:
            self.update_bae_dmm_names()
            for i in range(self.dmm_samples):
                bae_model = self.dmm.wrap(self._fit, self.bae_dmm_names[i], x_train,
                                               random_state=self.random_state + i)
        return bae_model

    def _fit(self, x_train, random_state=900):
        bae_set_seed(random_state)

        # =======AutoEncoder architecture
        if len(self.conv_architecture) > 0:
            encoder = Encoder([Conv1DLayers(input_dim=x_train.shape[2], conv_architecture=self.conv_architecture,
                                            conv_kernel=self.conv_kernel, activation="leakyrelu",
                                            conv_stride=self.conv_stride),
                               DenseLayers(architecture=self.dense_architecture, output_size=self.latent_dim,
                                           activation="leakyrelu",
                                           last_activation="leakyrelu")])
        else:
            encoder = Encoder([DenseLayers(architecture=self.dense_architecture[1:],
                                           input_size=self.dense_architecture[0],
                                           output_size=self.latent_dim, activation="leakyrelu",
                                           last_activation="leakyrelu")])
        decoder_mu = infer_decoder(encoder, last_activation='sigmoid')

        if self.likelihood != "hetero_gaussian":
            autoencoder = Autoencoder(encoder=encoder, decoder_mu=decoder_mu,
                                      homoscedestic_mode=self.homoscedestic_mode_map[self.likelihood])
        else:
            decoder_sig = infer_decoder(encoder, last_activation='none')
            autoencoder = Autoencoder(encoder=encoder, decoder_mu=decoder_mu, decoder_sig=decoder_sig)

        # ======training====
        bae_model = BAE_Ensemble(autoencoder=autoencoder,
                                 num_samples=self.bae_samples, learning_rate=self.learning_rate,
                                 homoscedestic_mode=self.homoscedestic_mode_map[self.likelihood],
                                 weight_decay=self.weight_decay,
                                 likelihood=self.likelihood_mode_map[self.likelihood], use_cuda=self.use_cuda)

        # pretrain to optimise reconstruction loss
        if self.train_model:
            train_loader = convert_dataloader(x_train)
            run_auto_lr_range_v2(train_loader, bae_model, run_full=True,
                                 supervised=False, window_size=5, num_epochs=1,
                                 mode="sigma" if bae_model.decoder_sigma_enabled else "mu", sigma_train="joint",
                                 plot=False)
            if not self.use_lr_finder:
                train_loader = x_train
            bae_model.fit(train_loader, num_epochs=self.num_epochs, supervised=False,
                          mode="sigma" if bae_model.decoder_sigma_enabled else "mu", sigma_train="joint")
            # bae_model.set_cuda(False)

            bae_model.model_name = bae_model.model_name + "_" + self.name
            # save_bae_model(bae_model)
        # else:
        #     bae_model = load_bae_model("BAE_Ensemble_" + self.name + ".p", folder="trained_models/")

        if self.posthoc_enable:
            self.set_forward_attribution_(bae_model)

        return bae_model

    def transform(self, message_data, key="quantities"):
        """
        Apply transform on every key of the quantities' dict if available
        Dict can have keys such as "test" , "ood", "train"
        Otherwise, directly compute on the quantities (assumed to be iterable)

        Each transformed data samples have a dict of {"nll_mu":nll_test_mu, "nll_var": nll_test_var, "y_var":y_test_var, "enc_var":encoded_test}

        e.g message_data["quantities"]["test"]["nll_mu"] OR message_data["quantities"]["nll_mu"]

        """

        if isinstance(message_data[key], dict):
            return {key_: self.transform_(message_data[key][key_]) if not self.use_dmm else
            self.dmm_transform_(message_data[key][key_]) for key_ in message_data[key].keys()}
        else:
            x_test = message_data[key]
            y_pred_test = self.transform_(x_test) if not self.use_dmm else self.dmm_transform_(x_test)
            return y_pred_test

    def dmm_transform_(self, x_test):
        y_pred_collect = []
        self.update_bae_dmm_names()
        for i in range(self.dmm_samples):
            bae_dmm_name = self._fit.__name__ + self.bae_dmm_names[i]
            if self.dmm.exist_model(bae_dmm_name):
                self.bae_model = self.dmm.load_model(datatype=bae_dmm_name)
            else:
                raise FileNotFoundError("BAE model not found by DMM. Have you trained the model yet?")
            y_pred_test = self.transform_(x_test)
            y_pred_collect.append(y_pred_test["y_pred"])
        y_pred_collect = np.concatenate(y_pred_collect)
        return y_pred_collect

    def transform_(self, x_test):
        """
        Internal transform function to compute NLL-MU, NLL-VAR, Y-PRED-VAR, and ENCODED-VAR
        """
        bae_model = self.bae_model

        # ===================predict==========================
        nll_key = "nll_sigma" if bae_model.decoder_sigma_enabled else "nll_homo"
        if self.likelihood == "bernoulli":
            nll_key = "bce"
        elif self.likelihood == "cbernoulli":
            nll_key = "cbce"


        if self.posthoc_enable:
            nll = np.array([get_attribution(autoencoder, input_data=convert_dataloader(x_test, batch_size=100),
                                            attr_class=self.posthoc_class) for autoencoder in bae_model.autoencoder])
            return {"y_pred":nll}

        elif self.return_samples:
            nll_test = bae_model.predict_samples(x_test, select_keys=[nll_key])
            # y_pred_test = bae_model.predict_samples(x_test, select_keys=["y_mu"])

            # return {"nll": nll_test, "y_pred": y_pred_test}
            # return {"y_pred": y_pred_test[:,0]}
            return {"y_pred": nll_test[:, 0]}

        else:
            nll_test = bae_model.predict_samples(x_test, select_keys=[nll_key])
            y_pred_test = bae_model.predict_samples(x_test, select_keys=["y_mu"])

            # compute statistics over BAE sampled parameters
            nll_test_mu = nll_test.mean(0)[0]
            nll_test_var = nll_test.var(0)[0]

            # get predictive uncertainty
            if bae_model.decoder_sigma_enabled:
                y_test_var = y_pred_test.var(0)[0] + bae_model.predict_samples(x_test, select_keys=["y_sigma"]).mean(0)[
                    0]
            elif bae_model.homoscedestic_mode == "every":
                y_test_var = y_pred_test.var(0)[0] + bae_model.get_homoscedestic_noise(return_mean=False)[0]
            else:
                y_test_var = y_pred_test.var(0)[0]

            # get encoded data
            encoded_test = bae_model.predict_latent(x_test, transform_pca=False)

            nll_test_mu = nll_test_mu.mean(-1)
            y_test_var = y_test_var.mean(-1)
            nll_test_var = nll_test_var.mean(-1)

            return {"nll_mu": nll_test_mu, "nll_var": nll_test_var, "y_var": y_test_var, "enc_var": encoded_test}

    def set_forward_attribution_(self, bae_model):
        """
        Set the forward method to that of a MSE. This is required to enable posthoc methods on the AE.
        """
        def forward(self, x):
            encoded = self.encoder(x)
            decoded_mu = self.decoder_mu(encoded)
            return ((decoded_mu - x) ** 2).mean(-1).mean(-1).reshape(-1, 1)

        for i in range(len(bae_model.autoencoder)):
            bae_model.autoencoder[i].forward = MethodType(forward, bae_model.autoencoder[i])

        return bae_model

class PropagateTransformAgent(ML_TransformAgent):
    """
    This agent expects to receive predictive samples (first dimension is dimension for BAE samples) from the BAE Agent, and apply fit_transform on every sample.

    In effect, we are constructing an ensemble of pipelines from the samples of BAE prediction.
    Note: the actual instantiation is done during fitting only to dynamically match the number of BAE samples sent in the `train` channel.


    """
    example_axis = 1

    def init_parameters(self, model=MultiMinMaxScaler,
                        random_state=123,
                        predict_train=True,
                        send_train_model=False,
                        use_dmm=False,
                        single_model=False,
                        propagate_key="y_pred",
                        return_mean=False,
                        send_test_samples=False,
                        example_axis=None,
                        func_apply_y=False,
                        **model_params):

        self.propagate_key = propagate_key
        self.single_model = single_model
        self.return_mean = return_mean
        self.send_test_samples = send_test_samples

        # create N models
        if model is not None:
            self.model_class = model

            # if it is a method which does not require fitting
            if inspect.ismethod(self.model_class) or inspect.isfunction(self.model_class):
                self.single_model = True  # override

        super(PropagateTransformAgent, self).init_parameters(model=None,
                                                             random_state=random_state,
                                                             predict_train=predict_train,
                                                             send_train_model=send_train_model,
                                                             use_dmm=use_dmm,
                                                             send_test_samples=send_test_samples,
                                                             example_axis=example_axis,
                                                             func_apply_y=func_apply_y,
                                                             **model_params
                                                             )

        if model is not None and (inspect.ismethod(self.model_class) or inspect.isfunction(self.model_class)):
            self.forward_model = self.instantiate_model()

    def instantiate_models(self, num_samples=5):
        model = [self.instantiate_model() for sample in range(num_samples)]
        return model

    def init_forward_model(self):
        if not self.single_model:
            self.forward_model = self.instantiate_models(num_samples=self.num_samples)
        else:
            self.forward_model = self.instantiate_model()

    def fit(self, message_data):
        """
        Fits self.model on message_data["quantities"]
        """
        # get number of samples
        self.num_samples = message_data["quantities"].shape[0]

        # if it is not a single model, here we instantiate multiple models as a list
        if not hasattr(self, "forward_model"):
            self.init_forward_model()

        # if the model is a list i.e not a single_model
        # then we fit every model in the list to the data
        if isinstance(self.forward_model, list):
            for sample_i, model in enumerate(self.forward_model):
                if hasattr(model, "fit"):
                    x_train = message_data["quantities"][sample_i]
                    self.forward_model[sample_i].fit(x_train, message_data["target"])

        # otherwise we fit on a single model on the mean of the data
        else:
            if hasattr(self.forward_model, "fit"):
                x_train_mean = message_data["quantities"].mean(0)
                self.forward_model.fit(x_train_mean, message_data["target"])
        return self.forward_model

    # def transform(self, message_data, keys):
    #     """
    #     Apply transform on every key of the quantities' dict if available
    #     Dict can have keys such as "test" , "ood", "train"
    #     Otherwise, directly compute on the quantities (assumed to be iterable)
    #
    #     Each transformed data samples have a dict of {"nll_mu":nll_test_mu, "nll_var": nll_test_var, "y_var":y_test_var, "enc_var":encoded_test}
    #
    #     e.g message_data["quantities"]["test"]["nll_mu"] OR message_data["quantities"]["nll_mu"]
    #
    #     """
    #
    #     if isinstance(message_data["quantities"],dict):
    #         return {key:self._transform(message_data["quantities"][key]) for key in message_data["quantities"].keys()}
    #     else:
    #         x_test = message_data["quantities"]
    #         y_pred_test = self._transform(x_test)
    #         return y_pred_test

    def _transform(self, message_data):
        """
        Internal function. Transforms and returns message_data["quantities"] using self.model
        """

        if not hasattr(self, "forward_model"):
            self.init_forward_model()

        # if model is a list of models, assume each model is mapped to the BAE i-th sample number
        if isinstance(self.forward_model, list):
            if hasattr(self.forward_model[0], "transform"):
                transformed_data = np.array(
                    [model.transform(message_data[sample_i]) for sample_i, model in enumerate(self.forward_model)])
            elif hasattr(self.forward_model[0], "predict"):
                transformed_data = np.array(
                    [model.predict(message_data[sample_i]) for sample_i, model in enumerate(self.forward_model)])
            else:
                transformed_data = np.array(
                    [model(message_data[sample_i]) for sample_i, model in enumerate(self.forward_model)])

        # otherwise, assume a single model, to be applied repeatedly on all the BAE samples
        else:
            if hasattr(self.forward_model, "transform"):
                transformed_data = np.array(
                    [self.forward_model.transform(message_data[sample_i]) for sample_i in range(message_data.shape[0])])
            elif hasattr(self.forward_model, "predict"):
                transformed_data = np.array(
                    [self.forward_model.predict(message_data[sample_i]) for sample_i in range(message_data.shape[0])])
            else:
                transformed_data = np.array(
                    [self.forward_model(message_data[sample_i]) for sample_i in range(message_data.shape[0])])

        # return mean of transformed data or not
        if self.return_mean:
            return transformed_data.mean(0)
        else:
            return transformed_data


class PropagateInverseTransformAgent(PropagateTransformAgent):
    """
    A special agent for performing propagated inverse transform on the BAE samples.
    Expected useful for inverse scaling transformers.
    """
    example_axis = 1

    def _transform(self, message_data):
        """
        Internal function. Inverse transforms and returns message_data["quantities"] using self.model
        """
        if isinstance(self.forward_model, list):
            transformed_data = np.array(
                [model.inverse_transform(message_data[sample_i]) for sample_i, model in enumerate(self.forward_model)])
        else:
            if hasattr(self.forward_model, "inverse_transform"):
                transformed_data = np.array(
                    [self.forward_model.inverse_transform(message_data[sample_i]) for sample_i in
                     range(message_data.shape[0])])

        if self.return_mean:
            return transformed_data.mean(0)
        else:
            return transformed_data


class PropagatePipelineAgent(PropagateTransformAgent):
    """
    This agent expects to receive reconstructed samples from the BAE Agent, and apply fit_transform on every sample.

    In effect, we are constructing an ensemble of pipelines from the samples of BAE prediction.
    """

    def init_parameters(self, pipeline_models=[], pipeline_params=[],
                        propagate_key="y_pred",
                        single_model=False,
                        predict_train=False,
                        send_train_model=False,
                        use_dmm=False,
                        return_mean=False,
                        send_test_samples=False,
                        example_axis=None,
                        random_state=123):
        """
        Initialise model parameters.
        Accepts either a class or a function as model.
        Stores the instantiated model in self.model.

        Parameters
        ----------
        model : class or function

        unsupervised : Boolean. Special keyword for handling fitting to 'target' in the self.fit function

        **model_params : keywords for model parameters instantiation.
        """
        # create N models
        self.pipeline_models = pipeline_models
        self.pipeline_params = pipeline_params

        super(PropagatePipelineAgent, self).init_parameters(model=None,
                                                            random_state=random_state,
                                                            predict_train=predict_train,
                                                            send_train_model=send_train_model,
                                                            single_model=single_model,
                                                            propagate_key=propagate_key,
                                                            return_mean=return_mean,
                                                            send_test_samples=send_test_samples,
                                                            example_axis=example_axis,
                                                            use_dmm=use_dmm)

    def instantiate_model(self):
        """
        Instantiate a single pipeline
        """
        new_model = make_pipeline(
            *[model(**model_param) for model, model_param in zip(self.pipeline_models, self.pipeline_params)])
        return new_model
