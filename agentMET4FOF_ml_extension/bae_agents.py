from pprint import pprint

from sklearn.pipeline import make_pipeline

from agentMET4FOF_ml_extension.ml_agents import ML_TransformAgent, ML_TransformPipelineAgent
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2

from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder, Autoencoder
from baetorch.baetorch.models.base_layer import DenseLayers, Conv1DLayers
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.data_model_manager import DataModelManager
from baetorch.baetorch.util.minmax import MultiMinMaxScaler
from baetorch.baetorch.util.misc import save_bae_model, load_bae_model
import numpy as np

# Convolutional BAE Agent
from baetorch.baetorch.util.seed import bae_set_seed


class CBAE_Agent(ML_TransformAgent):
    """
    Convolutional BAE Agent.

    total number of samples are actually use_dmm * dmm_samples

    """

    parameter_choices = {"first_nodes":[0,100,200,300,400,500],
                         "second_nodes":[0,100,200,300,400,500],
                         "latent_dim": [10,50,100],
                         "likelihood":["1_gaussian", "homo_gaussian", "hetero_gaussian", "bernoulli", "cbernoulli"],
                         "learning_rate":[0.1,0.01,0.001],
                         "bae_samples":[1,5,10],
                         # "use_cuda": [True,False],
                         # "train_model":[True,False],
                         "num_epochs":[10,50,100,150,200,250]
                         }

    def init_parameters(self, conv_architecture=[11, 5, 3],
                        dense_architecture=[100],
                        conv_kernel=[200, 100],
                        conv_stride=[10, 10],
                        latent_dim = 50,
                        likelihood="1_gaussian",
                        learning_rate =0.01,
                        bae_samples=1,
                        random_state=123,
                        use_cuda=True,
                        train_model=True,
                        send_train_model = False,
                        predict_train = True,
                        num_epochs=1,
                        use_dmm=False,
                        dmm_samples = 5,
                        return_samples = False,
                        **data_params):

        self.set_random_state(random_state)
        self.return_samples = return_samples
        self.send_train_model = send_train_model
        self.predict_train = predict_train
        self.model_params = {"conv_architecture":conv_architecture,
                        "dense_architecture":dense_architecture,
                        "conv_kernel":conv_kernel,
                        "conv_stride":conv_stride,
                        "latent_dim":latent_dim,
                        "likelihood":likelihood,
                        "bae_samples":bae_samples,
                        "num_epochs": num_epochs,
                        "use_dmm": use_dmm,
                        "dmm_samples": dmm_samples
                        }

        self.latent_dim = latent_dim
        self.dense_architecture = dense_architecture
        self.conv_architecture = conv_architecture
        self.conv_stride =conv_stride
        self.conv_kernel = conv_kernel
        self.likelihood = likelihood
        self.learning_rate = learning_rate
        self.homoscedestic_mode_map = {"1_gaussian":"none", "homo_gaussian":"every",
                                  "hetero_gaussian":"none",
                                  "bernoulli":"none",
                                  "cbernoulli":"none"}
        self.likelihood_mode_map = {"1_gaussian":"gaussian", "homo_gaussian":"gaussian",
                                  "hetero_gaussian":"gaussian",
                                  "bernoulli":"bernoulli",
                                  "cbernoulli":"cbernoulli"}
        self.bae_samples = bae_samples
        self.use_cuda = use_cuda
        self.train_model = train_model
        self.num_epochs= num_epochs
        self.dmm_samples = dmm_samples

        # if use Data Model Manager is enabled
        self.use_dmm = use_dmm
        if use_dmm:
           self.dmm = DataModelManager()

        # show model parameters on init
        pprint(self.model_params)

    def fit(self, message_data):
        """
        Checks whether DataModelManager is used.
        """
        x_train = message_data["quantities"].copy()

        # fit BAEs
        if not self.use_dmm:
                self.bae_model = self._fit(x_train, random_state=self.random_state)
        else:
            for i in range(self.dmm_samples):
                self.bae_model = self.dmm.wrap(self._fit,"cbae_fit"+str(self.model_params), x_train, random_state=self.random_state+i)
        return self.bae_model

    def _fit(self, x_train, random_state=900):
        bae_set_seed(random_state)

        # =======AutoEncoder architecture
        encoder = Encoder([Conv1DLayers(input_dim=x_train.shape[2], conv_architecture=self.conv_architecture,
                                        conv_kernel=self.conv_kernel, activation="leakyrelu", conv_stride=self.conv_stride),
                           DenseLayers(architecture=self.dense_architecture, output_size=self.latent_dim, activation="leakyrelu",
                                       last_activation="leakyrelu")])
        decoder_mu = infer_decoder(encoder, last_activation='sigmoid')

        if self.likelihood != "hetero_gaussian":
            autoencoder = Autoencoder(encoder=encoder, decoder_mu=decoder_mu, homoscedestic_mode=self.homoscedestic_mode_map[self.likelihood])
        else:
            decoder_sig = infer_decoder(encoder, last_activation='none')
            autoencoder = Autoencoder(encoder=encoder, decoder_mu=decoder_mu, decoder_sig=decoder_sig)

        # ======training====
        bae_model = BAE_Ensemble(autoencoder=autoencoder,
                                 num_samples=self.bae_samples, learning_rate=self.learning_rate,
                                 homoscedestic_mode=self.homoscedestic_mode_map[self.likelihood],
                                 likelihood=self.likelihood_mode_map[self.likelihood], use_cuda=self.use_cuda)

        # pretrain to optimise reconstruction loss
        if self.train_model:
            train_loader = convert_dataloader(x_train)
            run_auto_lr_range_v2(train_loader, bae_model, run_full=False,
                                 supervised=False, window_size=10, num_epochs=1,
                                 mode="sigma" if bae_model.decoder_sigma_enabled else "mu", sigma_train="joint",
                                 plot=False)
            bae_model.fit(train_loader, num_epochs=self.num_epochs, supervised=False,
                          mode="sigma" if bae_model.decoder_sigma_enabled else "mu", sigma_train="joint")
            bae_model.set_cuda(False)

            bae_model.model_name = bae_model.model_name + "_" + self.name
            save_bae_model(bae_model)
            self.bae_model = bae_model
        else:
            bae_model = load_bae_model("BAE_Ensemble_" + self.name + ".p", folder="trained_models/")
            self.bae_model = bae_model
        return bae_model

    def transform(self, message_data):
        """
        Apply transform on every key of the quantities' dict if available
        Dict can have keys such as "test" , "ood", "train"
        Otherwise, directly compute on the quantities (assumed to be iterable)

        Each transformed data samples have a dict of {"nll_mu":nll_test_mu, "nll_var": nll_test_var, "y_var":y_test_var, "enc_var":encoded_test}

        e.g message_data["quantities"]["test"]["nll_mu"] OR message_data["quantities"]["nll_mu"]

        """

        if isinstance(message_data["quantities"],dict):
            return {key:self.transform_(message_data["quantities"][key]) if not self.use_dmm else
            self.dmm_transform_(message_data["quantities"][key]) for key in message_data["quantities"].keys()}
        else:
            x_test = message_data["quantities"]
            y_pred_test = self.transform_(x_test) if not self.use_dmm else self.dmm_transform_(x_test)
            return y_pred_test

    def dmm_transform_(self, x_test):
        y_pred_collect = []

        for i in range(self.dmm_samples):
            self.bae_model = self.dmm.load_model(datatype="cbae_fit"+str(self.model_params), random_state=self.random_state+i)
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

        nll_test = bae_model.predict_samples(x_test, select_keys=[nll_key])
        y_pred_test = bae_model.predict_samples(x_test, select_keys=["y_mu"])

        if self.return_samples:
            # return {"nll": nll_test, "y_pred": y_pred_test}
            return {"y_pred": y_pred_test[:,0]}


        else:
            # compute statistics over BAE sampled parameters
            nll_test_mu = nll_test.mean(0)[0]
            nll_test_var = nll_test.var(0)[0]

            # get predictive uncertainty
            if bae_model.decoder_sigma_enabled:
                y_test_var = y_pred_test.var(0)[0] + bae_model.predict_samples(x_test, select_keys=["y_sigma"]).mean(0)[0]
            elif bae_model.homoscedestic_mode == "every":
                y_test_var = y_pred_test.var(0)[0] + bae_model.get_homoscedestic_noise(return_mean=False)[0]
            else:
                y_test_var = y_pred_test.var(0)[0]

            # get encoded data
            encoded_test = bae_model.predict_latent(x_test, transform_pca=False)

            nll_test_mu = nll_test_mu.mean(-1)
            y_test_var = y_test_var.mean(-1)
            nll_test_var = nll_test_var.mean(-1)

            return {"nll_mu":nll_test_mu, "nll_var": nll_test_var, "y_var":y_test_var, "enc_var":encoded_test}

class PropagateTransformAgent(ML_TransformAgent):
    """
    This agent expects to receive predictive samples (first dimension is dimension for BAE samples) from the BAE Agent, and apply fit_transform on every sample.

    In effect, we are constructing an ensemble of pipelines from the samples of BAE prediction.
    Note: the actual instantiation is done during fitting only to dynamically match the number of BAE samples sent in the `train` channel.

    """

    def init_parameters(self, model=MultiMinMaxScaler, random_state=123, predict_train=True,
                        send_train_model=False, use_dmm=False,
                        single_model = False, propagate_key="y_pred",
                        return_mean=False,
                        **model_params):
        # create N models
        self.model_class = model
        self.model_params = model_params

        self.propagate_key = propagate_key
        self.single_model = single_model
        self.return_mean = return_mean

        super(PropagateTransformAgent, self).init_parameters(model=None,
                                                             random_state=random_state,
                                                             predict_train=predict_train,
                                                             send_train_model=send_train_model,
                                                             use_dmm=use_dmm
                                                             )

    def instantiate_models(self, model, model_params={}, num_samples=5):
        model = [self.instantiate_model(model,model_params) for sample in range(num_samples)]
        return model

    def fit(self, message_data):
        """
        Fits self.model on message_data["quantities"]
        """
        # get number of samples
        self.num_samples = message_data["quantities"].shape[0]

        # if it is not a single model, here we instantiate multiple models as a list
        if not self.single_model:
            self.model = self.instantiate_models(model=self.model_class,
                                                model_params=self.model_params,
                                                num_samples=self.num_samples)
        else:
            self.model = self.instantiate_model(model=self.model_class,
                                                model_params=self.model_params)

        # if the model is a list i.e not a single_model
        # then we fit every model in the list to the data
        if isinstance(self.model, list):
            for sample_i,model in enumerate(self.model):
                if hasattr(model, "fit"):
                    x_train = message_data["quantities"][sample_i]
                    self.model[sample_i].fit(x_train, message_data["target"])

        # otherwise we fit on a single model on the mean of the data
        else:
            if hasattr(self.model, "fit"):
                x_train_mean = message_data["quantities"].mean(0)
                self.model.fit(x_train_mean, message_data["target"])
        return self.model

    def transform(self, message_data):
        """
        Apply transform on every key of the quantities' dict if available
        Dict can have keys such as "test" , "ood", "train"
        Otherwise, directly compute on the quantities (assumed to be iterable)

        Each transformed data samples have a dict of {"nll_mu":nll_test_mu, "nll_var": nll_test_var, "y_var":y_test_var, "enc_var":encoded_test}

        e.g message_data["quantities"]["test"]["nll_mu"] OR message_data["quantities"]["nll_mu"]

        """

        if isinstance(message_data["quantities"],dict):
            return {key:self._transform(message_data["quantities"][key]) for key in message_data["quantities"].keys()}
        else:
            x_test = message_data["quantities"]
            y_pred_test = self._transform(x_test)
            return y_pred_test


    def _transform(self, message_data):
        """
        Internal function. Transforms and returns message_data["quantities"] using self.model
        """

        # if model is a list of models, assume each model is mapped to the BAE i-th sample number
        if isinstance(self.model, list):
            if hasattr(self.model[0], "transform"):
                transformed_data = np.array([model.transform(message_data[sample_i]) for sample_i,model in enumerate(self.model)])
            elif hasattr(self.model[0], "predict"):
                transformed_data = np.array([model.predict(message_data[sample_i]) for sample_i,model in enumerate(self.model)])
            else:
                transformed_data = np.array([model(message_data[sample_i]) for sample_i,model in enumerate(self.model)])

        # otherwise, assume a single model, to be applied repeatedly on all the BAE samples
        else:
            if hasattr(self.model, "transform"):
                transformed_data = np.array([self.model.transform(message_data[sample_i]) for sample_i in range(message_data.shape[0])])
            elif hasattr(self.model, "predict"):
                transformed_data = np.array([self.model.predict(message_data[sample_i]) for sample_i in range(message_data.shape[0])])
            else:
                transformed_data = np.array([self.model(message_data[sample_i]) for sample_i in range(message_data.shape[0])])

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
    def _transform(self, message_data):
        """
        Internal function. Inverse transforms and returns message_data["quantities"] using self.model
        """
        if isinstance(self.model, list):
            transformed_data = np.array(
                [model.inverse_transform(message_data[sample_i]) for sample_i, model in enumerate(self.model)])
        else:
            if hasattr(self.model, "inverse_transform"):
                transformed_data = np.array([self.model.inverse_transform(message_data[sample_i]) for sample_i in range(message_data.shape[0])])
                
        if self.return_mean:
            return transformed_data.mean(0)
        else:
            return transformed_data


class PropagatePipelineAgent(PropagateTransformAgent):
    """
    This agent expects to receive reconstructed samples from the BAE Agent, and apply fit_transform on every sample.

    In effect, we are constructing an ensemble of pipelines from the samples of BAE prediction.
    """

    def init_parameters(self, models=[], model_params=[],
                        propagate_key = "y_pred",
                        single_model = False,
                        predict_train = False,
                        send_train_model=False,
                        use_dmm=False,
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
        self.model_classes = models
        self.model_params = model_params
        self.propagate_key = propagate_key

        super(PropagatePipelineAgent, self).init_parameters(models=None,
                                                            model_params=None,
                                                            random_state=random_state,
                                                            predict_train=predict_train,
                                                            send_train_model=send_train_model,
                                                            use_dmm=use_dmm)


    def instantiate_model(self, models, model_params={}):
        new_model = make_pipeline(*[model(**model_param) for model, model_param in zip(models, model_params)])
        return new_model
    
    def instantiate_models(self, model, model_params={}, num_samples=5):
        model = [self.instantiate_model(model,model_params) for sample in range(num_samples)]
        return model



