import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix, matthews_corrcoef

def predict_drift_labels(sensor_nll, k:int):
    """
    Predict 1 (drift) for top-k sensors and the rest as 0 (non-drift)

    Parameters
    ----------
    sensor_nll : np.ndarray
        Sensor explanations of NLL

    k : int
        Top-k sensors explanations are assigned as drifting
    """
    total_sensors = sensor_nll.shape[-1]

    # sensor_argsort = np.argsort(np.argsort(sensor_nll,axis=1),axis=1)
    sensor_argsort = rankdata(sensor_nll,axis=1, method="ordinal").astype(int)-1
    topk_sensors = sensor_argsort >= (total_sensors-k)

    # create the assigned drifting labels
    drifting_labels = sensor_argsort.copy()*0
    drifting_labels[topk_sensors] += 1

    return drifting_labels


def get_true_labels(sensor_nll, perturbed_sensors : list):
    total_examples = sensor_nll.shape[0]
    total_sensors = sensor_nll.shape[-1]

    # true labels for sensors which are drifting
    true_labels = [1 if sensor_i in perturbed_sensors else 0 for sensor_i in range(total_sensors)]
    true_labels = np.repeat(np.expand_dims(true_labels,axis=0), total_examples, axis=0)
    return true_labels

def calc_gmean_sser(sensor_nll,perturbed_sensors):
    predicted_labels = predict_drift_labels(sensor_nll, k=len(perturbed_sensors))
    true_labels = get_true_labels(sensor_nll, perturbed_sensors)

    # now compare true labels and assigned labels
    tn, fp, fn, tp = confusion_matrix(true_labels.flatten(), predicted_labels.flatten()).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    gmean_sser = np.sqrt(sensitivity*specificity)

    return {"sensitivity":sensitivity,"specificity":specificity,"gmean_sser":gmean_sser}

def calc_mcc(sensor_nll,perturbed_sensors):
    predicted_labels = predict_drift_labels(sensor_nll, k=len(perturbed_sensors))
    true_labels = get_true_labels(sensor_nll, perturbed_sensors)

    matthew_corr = matthews_corrcoef(true_labels.flatten(), predicted_labels.flatten())

    return matthew_corr
