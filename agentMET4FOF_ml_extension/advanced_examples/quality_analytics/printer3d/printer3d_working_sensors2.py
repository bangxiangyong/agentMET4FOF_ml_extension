import dill
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from agentMET4FOF_ml_extension.advanced_examples.condition_monitoring.unsupervised_zema_emc_bae_v5 import moving_average
from agentMET4FOF_ml_extension.advanced_examples.quality_analytics.anomaly_thresholder import AnomalyThreshold
from agentMET4FOF_ml_extension.util import calc_auroc
from agentMET4FOF_ml_extension.util.calc_auroc import calc_auroc_score
from agentMET4FOF_ml_extension.util.resample_sensor import apply_along_sensor
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder, Autoencoder
from baetorch.baetorch.models.base_layer import ConvLayers, DenseLayers, Conv1DLayers
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.minmax import MultiMinMaxScaler
from baetorch.baetorch.util.seed import bae_set_seed
from baetorch.baetorch.util.misc import save_bae_model, load_bae_model

import matplotlib
matplotlib.use("qt5agg")
# bae_set_seed(4567)
bae_set_seed(321)
# bae_set_seed(1233)
# bae_set_seed(78979)
# parsed_data_file = "printer3d_parsed.p"

def minmax_scale_local(x):
    x_max = np.max(x)
    x_min = np.min(x)
    x_scaled = (x-x_min)/(x_max-x_min)
    return x_scaled

parsed_data_file = "parsed_fft_data_file.p"
parsed_data = dill.load(open(parsed_data_file,"rb"))

sensors_data = parsed_data["fft_sensors"]
y_labels = parsed_data["targets"]

healthy_arg = np.argwhere(y_labels<1).reshape(-1)
ood_arg = np.argwhere(y_labels>=1).reshape(-1)

# train test split
test_size = 0.5

x_train, x_test , y_train, y_test = train_test_split(sensors_data[healthy_arg],
                                                               y_labels[healthy_arg], test_size=test_size)
x_ood = sensors_data[ood_arg].copy()
y_ood = y_labels[ood_arg]

# remove the first
x_train = x_train[:,1:]
x_test = x_test[:,1:]
x_ood = x_ood[:,1:]

# remove the last
# x_train = x_train[:,:-1]
# x_test = x_test[:,:-1]
# x_ood = x_ood[:,:-1]

# apply min max scaler
# scaler = MultiMinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# x_ood = scaler.transform(x_ood)

# apply local min max
x_train = apply_along_sensor(x_train, func=minmax_scale_local, sensor_axis=2)
x_test = apply_along_sensor(x_test, func=minmax_scale_local, sensor_axis=2)
x_ood = apply_along_sensor(x_ood, func=minmax_scale_local, sensor_axis=2)


# select sensors
# select_sensor = 0
# x_test= np.expand_dims(x_test[:,:,select_sensor],-1)
# x_ood= np.expand_dims(x_ood[:,:,select_sensor],-1)
# x_train= np.expand_dims(x_train[:,:,select_sensor],-1)

# move axis
x_test= np.moveaxis(x_test,1,2)
x_ood= np.moveaxis(x_ood,1,2)
x_train= np.moveaxis(x_train,1,2)



# model architecture
latent_dim = 50
input_channel = x_train.shape[1]
use_cuda = True
conv_architecture=[input_channel,10,15,25]
# conv_architecture=[input_channel,15]
train_model = True
use_lr_finder = True
num_epochs = 50
bae_samples = 10


#specify encoder
#with convolutional layers and hidden dense layer
encoder = Encoder([Conv1DLayers(input_dim=x_train.shape[2], conv_architecture=conv_architecture,
                                            conv_kernel=[12,6,2], activation="leakyrelu",
                                            conv_stride=[2,2,2]),
           DenseLayers(architecture=[100],output_size=latent_dim)])

# encoder = Encoder([Conv1DLayers(input_dim=x_train.shape[2], conv_architecture=conv_architecture,
#                                             conv_kernel=[12,], activation="leakyrelu",
#                                             conv_stride=[2]),
#            DenseLayers(architecture=[1000],output_size=latent_dim)])


#specify decoder-mu
decoder_mu = infer_decoder(encoder,last_activation="sigmoid") #symmetrical to encoder

#combine them into autoencoder
autoencoder = Autoencoder(encoder, decoder_mu)

#convert into BAE
# bae_model = BAE_Ensemble(autoencoder=autoencoder, use_cuda=use_cuda,
#                          anchored=True, weight_decay=0.001,
#                          likelihood="gaussian",
#                          homoscedestic_mode ="every",
#                          num_samples=bae_samples)
bae_model = BAE_Ensemble(autoencoder=autoencoder, use_cuda=use_cuda,
                         anchored=True, weight_decay=0.0005,
                         likelihood="cbernoulli",
                         num_samples=bae_samples)
if train_model:
    train_loader = convert_dataloader(x_train)
    run_auto_lr_range_v2(train_loader, bae_model, run_full=True,
                         supervised=False, window_size=3, num_epochs=10,
                         mode="sigma" if bae_model.decoder_sigma_enabled else "mu", sigma_train="joint",
                         plot=False)
    bae_model.fit(train_loader, num_epochs=num_epochs, supervised=False,
                  mode="sigma" if bae_model.decoder_sigma_enabled else "mu", sigma_train="joint")
    # save_bae_model(bae_model)
    # bae_model.set_cuda(use_cuda)
else:
    bae_model = load_bae_model("BAE_Ensemble.p",
                               folder="trained_models\\")

    bae_model.set_cuda(use_cuda)
    print("DONE LOADING BAE MODEL...")

# y_train_samples = bae_model.predict_samples(x_train, select_keys=["y_mu","nll_homo"])
# y_test_samples = bae_model.predict_samples(x_test, select_keys=["y_mu","nll_homo"])
# y_ood_samples = bae_model.predict_samples(x_ood, select_keys=["y_mu","nll_homo"])

y_train_samples = bae_model.predict_samples(x_train, select_keys=["y_mu","se"])
y_test_samples = bae_model.predict_samples(x_test, select_keys=["y_mu","se"])
y_ood_samples = bae_model.predict_samples(x_ood, select_keys=["y_mu","se"])


y_train_nll_mean = y_train_samples.mean(0)[1].sum(-1).sum(-1)
y_test_nll_mean = y_test_samples.mean(0)[1].sum(-1).sum(-1)
y_ood_nll_mean = y_ood_samples.mean(0)[1].sum(-1).sum(-1)

# select_sensor = 0
# y_train_nll_mean = y_train_samples.mean(0)[1].sum(-1)[:,select_sensor]
# y_test_nll_mean = y_test_samples.mean(0)[1].sum(-1)[:,select_sensor]
# y_ood_nll_mean = y_ood_samples.mean(0)[1].sum(-1)[:,select_sensor]

# y_train_nll_mean = (y_train_samples.mean(0)[1]/y_train_samples.var(0)[1]).sum(-1).sum(-1)
# y_test_nll_mean = (y_test_samples.mean(0)[1]/y_test_samples.var(0)[1]).sum(-1).sum(-1)
# y_ood_nll_mean = (y_ood_samples.mean(0)[1]/y_ood_samples.var(0)[1]).sum(-1).sum(-1)

# y_train_nll_mean = (y_train_samples.mean(0)[1]/y_train_samples.var(0)[1]).mean(-1).mean(-1)
# y_test_nll_mean = (y_test_samples.mean(0)[1]/y_test_samples.var(0)[1]).mean(-1).mean(-1)
# y_ood_nll_mean = (y_ood_samples.mean(0)[1]/y_ood_samples.var(0)[1]).mean(-1).mean(-1)

# y_train_nll_mean = (y_train_samples.sum(-1).sum(-1).mean(0)[1]/y_train_samples.sum(-1).sum(-1).var(0)[1])
# y_test_nll_mean = (y_test_samples.sum(-1).sum(-1).mean(0)[1]/y_test_samples.sum(-1).sum(-1).var(0)[1])
# y_ood_nll_mean = (y_ood_samples.sum(-1).sum(-1).mean(0)[1]/y_ood_samples.sum(-1).sum(-1).var(0)[1])


# y_train_nll_var= y_train_samples.sum(-1).sum(-1).var(0)[1]
# y_test_nll_var = y_test_samples.sum(-1).sum(-1).var(0)[1]
# y_ood_nll_var= y_ood_samples.sum(-1).sum(-1).var(0)[1]

y_train_nll_var= y_train_samples.var(0)[1].sum(-1).sum(-1)
y_test_nll_var = y_test_samples.var(0)[1].sum(-1).sum(-1)
y_ood_nll_var= y_ood_samples.var(0)[1].sum(-1).sum(-1)

y_train_yvar= y_train_samples.var(0)[0].sum(-1).sum(-1)
y_test_yvar = y_test_samples.var(0)[0].sum(-1).sum(-1)
y_ood_yvar= y_ood_samples.var(0)[0].sum(-1).sum(-1)


auroc_nll_mean = calc_auroc_score(y_test_nll_mean,y_ood_nll_mean)
auroc_nll_var = calc_auroc_score(y_test_nll_var,y_ood_nll_var)
auroc_yvar = calc_auroc_score(y_test_yvar,y_ood_yvar)

print(auroc_nll_mean)
print(auroc_nll_var)
print(auroc_yvar)

fig, ((ax1,ax2),(ax3,ax4), (ax5,ax6)) = plt.subplots(3,2)
sns.kdeplot(y_train_nll_mean, ax=ax1)
sns.kdeplot(y_test_nll_mean, ax=ax1)
sns.kdeplot(y_ood_nll_mean, ax=ax1)

sns.ecdfplot(y_train_nll_mean, ax=ax2)
sns.ecdfplot(y_test_nll_mean, ax=ax2)
sns.ecdfplot(y_ood_nll_mean, ax=ax2)

sns.kdeplot(y_train_nll_var, ax=ax3)
sns.kdeplot(y_test_nll_var, ax=ax3)
sns.kdeplot(y_ood_nll_var, ax=ax3)

sns.ecdfplot(y_train_nll_var, ax=ax4)
sns.ecdfplot(y_test_nll_var, ax=ax4)
sns.ecdfplot(y_ood_nll_var, ax=ax4)

sns.kdeplot(y_train_yvar, ax=ax5)
sns.kdeplot(y_test_yvar, ax=ax5)
sns.kdeplot(y_ood_yvar, ax=ax5)

sns.ecdfplot(y_train_yvar, ax=ax6)
sns.ecdfplot(y_test_yvar, ax=ax6)
sns.ecdfplot(y_ood_yvar, ax=ax6)


plt.tight_layout()

anomaly_thresholder = AnomalyThreshold(threshold=0.75)
anomaly_thresholder.fit(y_train_nll_mean)

y_binary_test = anomaly_thresholder.transform(y_test_nll_mean)
y_binary_ood = anomaly_thresholder.transform(y_ood_nll_mean)

y_binary_pred = np.concatenate((y_binary_test,y_binary_ood))
y_binary_true = np.concatenate((np.zeros_like(y_binary_test),np.ones_like(y_binary_ood)))


mcc_score = matthews_corrcoef(y_binary_true, y_binary_pred)
f1_scores = f1_score(y_binary_true, y_binary_pred)


plt.figure()
plt.scatter(y_train_nll_mean,y_train_nll_var)
plt.scatter(y_test_nll_mean,y_test_nll_var)
plt.scatter(y_ood_nll_mean,y_ood_nll_var)





# plt.hist(y_train_nll_mean, density=True)
# plt.hist(y_test_nll_mean, density=True)
# plt.hist(y_ood_nll_mean, density=True)

# sns.kdeplot(y_train_nll_var)
# sns.kdeplot(y_test_nll_var)
# sns.kdeplot(y_ood_nll_var)

# plot some outputs
plt.figure()
plt.plot(y_train_samples[:,0][0,0,0])
plt.plot(y_train_samples[:,0][1,0,0])
plt.plot(y_train_samples[:,0][2,0,0])
plt.plot(y_train_samples[:,0][3,0,0])
plt.plot(y_train_samples[:,0][4,0,0])


plt.figure()
plt.plot(x_train[0,0])