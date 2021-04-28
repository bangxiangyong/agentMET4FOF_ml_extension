import torch
from captum.attr import DeepLift, ShapleyValueSampling
import numpy as np

def _get_attribution(autoencoder, input_data, attr_class = DeepLift):
    if isinstance(input_data, np.ndarray):
        input_data_tensor = torch.from_numpy(input_data).float()
    else:
        input_data_tensor = input_data
    baseline = torch.zeros_like(input_data_tensor).float()
    if autoencoder.use_cuda:
        baseline = baseline.cuda()
    exp_model = attr_class(autoencoder)
    if attr_class == ShapleyValueSampling:
        attributions = exp_model.attribute(input_data_tensor, baseline, target=0, n_samples=10)
    else:
        attributions = exp_model.attribute(input_data_tensor, baseline, target=0, return_convergence_delta=False)
    attributions = attributions.detach().cpu().numpy()

    return attributions

def get_attribution(autoencoder, input_data, attr_class = DeepLift):
    """
    Handles assigning attribution on dataloader and single numpy array separately.
    """
    if isinstance(input_data, torch.utils.data.dataloader.DataLoader):
        attribution_list = []
        for batch_idx, (data, target) in enumerate(input_data):
            if autoencoder.use_cuda:
                data = data.cuda()
            attribution_list.append(_get_attribution(autoencoder=autoencoder, input_data=data, attr_class=attr_class))
        return np.concatenate(attribution_list, axis=0)
    else:
        return _get_attribution(autoencoder=autoencoder, input_data=input_data, attr_class=attr_class)


