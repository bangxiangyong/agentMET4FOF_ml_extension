import dill
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from agentMET4FOF_ml_extension.advanced_examples.condition_monitoring.unsupervised_zema_emc_bae_v5 import moving_average
from agentMET4FOF_ml_extension.util import calc_auroc
from agentMET4FOF_ml_extension.util.calc_auroc import calc_auroc_score
from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v2
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder, Autoencoder
from baetorch.baetorch.models.base_layer import ConvLayers, DenseLayers
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed



from baetorch.baetorch.util.misc import save_bae_model, load_bae_model
import matplotlib
matplotlib.use("qt5agg")
bae_set_seed(1233)
# parsed_data_file = "printer3d_parsed.p"
parsed_data_file = "printer3d_parsed_10color.p"
parsed_data = dill.load(open(parsed_data_file,"rb"))

videos = np.array(parsed_data["videos"], dtype="object")
y_labels = parsed_data["targets"]

healthy_arg = np.argwhere(y_labels<1).reshape(-1)
ood_arg = np.argwhere(y_labels>=1).reshape(-1)

# train test split
test_size = 0.5
videos_train, videos_test , y_train, y_test = train_test_split(videos[healthy_arg],
                                                               y_labels[healthy_arg], test_size=test_size)
videos_ood = videos[ood_arg]
y_ood = y_labels[ood_arg]

x_train = np.concatenate(videos_train)
# x_train = np.concatenate(videos_train)
# x_train = np.concatenate(videos_train)
# x_train = np.concatenate(videos_train)

#=== prepare bae===

#model architecture
latent_dim = 50
input_dim = (42,128)
input_channel = 3
use_cuda = True
conv_architecture=[input_channel,20,25]
train_model = True
use_lr_finder = True
num_epochs = 2
bae_samples = 5

#specify encoder
#with convolutional layers and hidden dense layer
encoder = Encoder([ConvLayers(input_dim=input_dim,conv_architecture=conv_architecture,
                              conv_kernel=[(4,12),(2,4)],
                              conv_stride=[2,2],
                              last_activation="sigmoid"),
           DenseLayers(architecture=[100],output_size=latent_dim)])

#specify decoder-mu
decoder_mu = infer_decoder(encoder,last_activation="sigmoid") #symmetrical to encoder

#combine them into autoencoder
autoencoder = Autoencoder(encoder, decoder_mu)

#convert into BAE
bae_model = BAE_Ensemble(autoencoder=autoencoder, use_cuda=use_cuda,
                         anchored=True, weight_decay=0.008,
                         likelihood="gaussian",
                         homoscedestic_mode ="every",
                         num_samples=bae_samples)

if train_model:
    train_loader = convert_dataloader(np.moveaxis(x_train, 3,1))
    run_auto_lr_range_v2(train_loader, bae_model, run_full=True,
                         supervised=False, window_size=5, num_epochs=1,
                         mode="sigma" if bae_model.decoder_sigma_enabled else "mu", sigma_train="joint",
                         plot=False)

    bae_model.fit(train_loader, num_epochs=num_epochs, supervised=False,
                  mode="sigma" if bae_model.decoder_sigma_enabled else "mu", sigma_train="joint")
    # save_bae_model(bae_model)
    bae_model.set_cuda(use_cuda)
else:
    bae_model = load_bae_model("BAE_Ensemble.p",
                               folder="trained_models\\")

    bae_model.set_cuda(use_cuda)
    print("DONE LOADING BAE MODEL...")
#===============predicting==========

def predict_bae(bae_model, x_train, return_sum=False):

    y_pred_samples = bae_model.predict_samples(np.moveaxis(x_train, 3,1), select_keys=["y_mu","nll_homo"])
    y_mu_mean = y_pred_samples.mean(0)[0]
    y_mu_var = y_pred_samples.var(0)[0]
    y_nll_mean =y_pred_samples.mean(0)[1]
    y_nll_var = y_pred_samples.var(0)[1]

    y_mu_mean = np.moveaxis(y_mu_mean, 1,3)
    y_mu_var = np.moveaxis(y_mu_var, 1,3)
    y_nll_mean = np.moveaxis(y_nll_mean, 1,3)
    y_nll_var = np.moveaxis(y_nll_var, 1,3)

    if return_sum:
        while len(y_mu_mean.shape)>1:
            y_mu_mean = y_mu_mean.mean(-1)
            y_mu_var = y_mu_var.mean(-1)
            y_nll_mean = y_nll_mean.mean(-1)
            y_nll_var = y_nll_var.mean(-1)

    return {"y_mu_mean":y_mu_mean,
            "y_mu_var":y_mu_var,
            "y_nll_mean":y_nll_mean,
            "y_nll_var":y_nll_var
            }

sample_i =1
x_test = videos_test[0]
x_ood = videos_ood[0]

y_pred_train = predict_bae(bae_model, x_train[-60:])
# y_pred_test = predict_bae(bae_model, x_test)
# y_pred_ood = predict_bae(bae_model, x_ood)

y_input = x_train[sample_i]

fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1)
ax1.imshow(y_input, cmap=plt.cm.binary)
ax2.imshow(y_pred_train["y_mu_mean"][sample_i])
ax3.imshow(y_pred_train["y_mu_var"][sample_i].mean(-1), cmap=plt.cm.binary)
ax4.imshow(y_pred_train["y_nll_mean"][sample_i].mean(-1), cmap=plt.cm.binary)
ax5.imshow(y_pred_train["y_nll_var"][sample_i].mean(-1), cmap=plt.cm.binary)
plt.tight_layout()
plt.show()

#===============AUROC==========================
x_test_i = videos_test[0]
x_ood_i = videos_ood[0]

# y_pred_test_i = predict_bae(bae_model, x_test_i, return_sum=True)
# y_pred_ood_i = predict_bae(bae_model, x_ood_i, return_sum=True)

def get_nll(bae_model, videos, start_time=10, return_mean=False):
    y_mu_var = []
    y_nll_mean = []
    y_nll_var = []

    for x_i in videos:
        y_pred_test_i = predict_bae(bae_model, x_i, return_sum=True)
        y_mu_var_i = y_pred_test_i["y_mu_var"][start_time:]
        y_nll_mean_i = y_pred_test_i["y_nll_mean"][start_time:]
        y_nll_var_i = y_pred_test_i["y_nll_var"][start_time:]

        if return_mean:
            y_mu_var_i = y_mu_var_i.mean()
            y_nll_mean_i = y_nll_mean_i.mean()
            y_nll_var_i = y_nll_var_i.mean()

        y_mu_var.append(y_mu_var_i)
        y_nll_mean.append(y_nll_mean_i)
        y_nll_var.append(y_nll_var_i)

    return {"y_mu_var": y_mu_var,
            "y_nll_mean": y_nll_mean,
            "y_nll_var": y_nll_var}

start_time = 10
y_pred_train_mean = get_nll(bae_model, videos_train, start_time=start_time, return_mean=True)
y_pred_test_mean = get_nll(bae_model, videos_test, start_time=start_time, return_mean=True)
y_pred_ood_mean = get_nll(bae_model, videos_ood, start_time=start_time, return_mean=True)


fig, (ax1,ax2,ax3) = plt.subplots(3,1)
ax1.hist(y_pred_train_mean["y_mu_var"], density=True)
ax1.hist(y_pred_test_mean["y_mu_var"], density=True)
ax1.hist(y_pred_ood_mean["y_mu_var"], density=True)
ax2.hist(y_pred_train_mean["y_nll_mean"], density=True)
ax2.hist(y_pred_test_mean["y_nll_mean"], density=True)
ax2.hist(y_pred_ood_mean["y_nll_mean"], density=True)
ax3.hist(y_pred_train_mean["y_nll_var"], density=True)
ax3.hist(y_pred_test_mean["y_nll_var"], density=True)
ax3.hist(y_pred_ood_mean["y_nll_var"], density=True)
ax1.legend(["TRAIN","TEST","OOD"])


y_pred_train = get_nll(bae_model, videos_train, start_time=start_time, return_mean=False)
y_pred_test = get_nll(bae_model, videos_test, start_time=start_time, return_mean=False)
y_pred_ood = get_nll(bae_model, videos_ood, start_time=start_time, return_mean=False)

plt.figure()
for y_pred in y_pred_train["y_mu_var"]:
    plt.plot(y_pred, color="tab:blue")
for y_pred in y_pred_test["y_mu_var"]:
    plt.plot(y_pred, color="tab:orange")
for y_pred in y_pred_ood["y_mu_var"]:
    plt.plot(y_pred, color="tab:green")

# y_pred_test_i["y_nll_mean"][10:].mean()
# y_pred_ood_i["y_nll_mean"][10:].mean()

import seaborn as sns

plt.figure()
sns.kdeplot(y_pred_train_mean["y_mu_var"])
sns.kdeplot(y_pred_test_mean["y_mu_var"])
sns.kdeplot(y_pred_ood_mean["y_mu_var"])

plt.figure()
sns.kdeplot(y_pred_train_mean["y_nll_var"])
sns.kdeplot(y_pred_test_mean["y_nll_var"])
sns.kdeplot(y_pred_ood_mean["y_nll_var"])


plt.figure()
sns.ecdfplot(y_pred_train_mean["y_mu_var"])
sns.ecdfplot(y_pred_test_mean["y_mu_var"])
sns.ecdfplot(y_pred_ood_mean["y_mu_var"])

print(calc_auroc_score(y_pred_test_mean["y_mu_var"], y_pred_ood_mean["y_mu_var"]))
print(calc_auroc_score(y_pred_test_mean["y_nll_mean"], y_pred_ood_mean["y_nll_mean"]))
print(calc_auroc_score(y_pred_test_mean["y_nll_var"],y_pred_ood_mean["y_nll_var"]))
print(calc_auroc_score(y_pred_test_mean["y_nll_var"],y_pred_ood_mean["y_nll_var"]))

#=====LATENT SPACE======#
def predict_latent(bae_model, x_i, window_size=250, normalise_0=True, cut_start=50):
    y_latent_test_i = bae_model.predict_latent(np.moveaxis(x_i, 3,1),transform_pca=False)

    y_latent_test_i_mean = y_latent_test_i[0]
    y_latent_test_i_var = y_latent_test_i[1]

    # y_latent_test_i_coord0 = y_latent_test_i_mean[:,:25].mean(-1)
    # y_latent_test_i_coord1 = y_latent_test_i_mean[:,25:].mean(-1)

    y_latent_test_i_coord0 = (y_latent_test_i_mean[:,:25]/y_latent_test_i_var[:,:25]).mean(-1)
    y_latent_test_i_coord1 = (y_latent_test_i_mean[:,25:]/y_latent_test_i_var[:,25:]).mean(-1)

    # y_latent_test_i_coord0 = (y_latent_test_i_mean[:,:25]).mean(-1)
    # y_latent_test_i_coord1 = (y_latent_test_i_mean[:,25:]).mean(-1)

    # y_latent_test_i_coord0 = (y_latent_test_i_var[:,:25]).mean(-1)
    # y_latent_test_i_coord1 = (y_latent_test_i_var[:,25:]).mean(-1)

    y_latent_test_i_coord0 = moving_average(y_latent_test_i_coord0,window_size=window_size).reshape(-1,1)
    y_latent_test_i_coord1 = moving_average(y_latent_test_i_coord1,window_size=window_size).reshape(-1,1)

    y_latent_test_i_coord0 = y_latent_test_i_coord0[cut_start:]
    y_latent_test_i_coord1 = y_latent_test_i_coord1[cut_start:]

    if normalise_0:
        y_latent_test_i_coord0 = y_latent_test_i_coord0 - y_latent_test_i_coord0[0]
        y_latent_test_i_coord1 = y_latent_test_i_coord1 - y_latent_test_i_coord1[0]

    return np.concatenate((y_latent_test_i_coord0, y_latent_test_i_coord1),axis=1)

window_size = 50
y_larent_train = [predict_latent(bae_model, x_i, window_size=window_size) for x_i in videos_train]
y_larent_test = [predict_latent(bae_model, x_i, window_size=window_size) for x_i in videos_test]
y_larent_ood = [predict_latent(bae_model, x_i, window_size=window_size) for x_i in videos_ood]

fig = plt.figure()
for latent_train_i in y_larent_train:
    plt.scatter(latent_train_i[:, 0], latent_train_i[:, 1],color="tab:green",alpha=0.5)
for latent_test_i in y_larent_test:
    plt.scatter(latent_test_i[:, 0], latent_test_i[:, 1],color="tab:blue",alpha=0.5)
for latent_ood_i in y_larent_ood:
    plt.scatter(latent_ood_i[:, 0], latent_ood_i[:, 1],color="tab:red",alpha=0.5)














