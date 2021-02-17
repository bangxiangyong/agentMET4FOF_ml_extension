from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Pearson_FeatureSelection():
    def __init__(self, n_of_features=500):
        self.sensor_indices = []
        self.feature_indices = []

        # Defining how much of features with biggest Pearson correllation coeff. will be selected.
        # "How many features out of %s you want to select (recommended is 500): " % n_features_for_select)
        self.n_of_features = n_of_features

    def fit(self, x_data, y_data):
        sorted_values_from_all_sensors = x_data
        n_sensors = len(sorted_values_from_all_sensors)
        n_input_features = sorted_values_from_all_sensors[0].shape[1]  # 100

        n_features_for_select = 0
        for i in range(len(sorted_values_from_all_sensors)):
            n_features_for_select = n_features_for_select + int(len(sorted_values_from_all_sensors[i].iloc[0][:]))

        target_matrix = y_data[:,0] if len(y_data.shape) > 1 else y_data

        # print("\nDimension of target matrix is:")
        # print("                                                 ", target_matrix.shape)
        # print("Dimension of amplitude matrix for one sensor is:")
        # print("                                                 ", sorted_values_from_all_sensors[0].iloc[:][:].shape)

        corr = list(range(n_sensors))  # Making list for correlation coefficients.
        p_value = list(range(n_sensors))

        for j in range(n_sensors):  # Making sublists in "corr" for each sensor.
            corr[j] = list(range(n_input_features))
            p_value[j] = list(range(n_input_features))

        # Calculating correlation coefficients for each column of each sensor with respect to target.
        for j in range(n_sensors):
            for i in range(n_input_features):
                corr[j][i], p_value[j][i] = pearsonr(np.abs(sorted_values_from_all_sensors[j].iloc[:][i]),
                                                     target_matrix)
        # matrix_corr_coeff = np.transpose(pd.DataFrame(corr))# Transforming list of correlation coefficients to data frame.
        corr_array = np.array(corr)  # Transforming list of correlation coefficients to nparray

        # print("Array of correlation coefficients has size:")
        # print("                                                 ",corr_array.shape)

        def largest_indices(array, n):  # Function that find indices for 500 biggest Pearson-
            """Returns the n largest indices from a numpy array."""  # -correlation coefficients.
            flat = array.flatten()
            indices = np.argpartition(flat, -n)[-n:]
            indices = indices[np.argsort(-flat[indices])]
            return np.unravel_index(indices, array.shape)

        # sensor_indices is the index of the sensor number.
        # feature_indices is the index of the feature number for each sensor number.
        sensor_indices, feature_indices = largest_indices(corr_array, self.n_of_features)

        # print("Sensor indices of location of features in >sorted_values_from_all_sensors< matrix: \n")
        # print(sensor_indices)
        # print("\nColumn indices of location of features in >sorted_values_from_all_sensors< matrix: \n")
        # print(feature_indices)
        self.sensor_indices = sensor_indices
        self.feature_indices = feature_indices
        return self

    def fit_transform(self, x_data, y_data):
        self.fit(x_data, y_data)
        abs_top_n_together_matrix = self.transform(x_data)
        return abs_top_n_together_matrix

    def transform(self, x_data):
        # Initialising a list of best features. 11 sublists containing features from each sensor, respectively.
        sorted_values_from_all_sensors = x_data
        n_sensors = len(sorted_values_from_all_sensors)
        top_n_features = [[] for n in range(n_sensors)]
        # NOTE: top_n_features =[[]]*int(n_sensors) doesn't work !!!

        sensor_n = self.sensor_indices
        for i in range(n_sensors):
            for j in range(len(self.sensor_indices)):
                if self.sensor_indices[j] == i:
                    top_n_features[i].append(sorted_values_from_all_sensors[i].iloc[:][self.feature_indices[j]]);

        for i in range(n_sensors):
            for j in range(len(top_n_features[i])):
                top_n_features[i][j] = list(top_n_features[i][j])

        # Merging sublists into one list with all elements.
        top_n_together = [j for i in top_n_features for j in i]

        top_n_together_matrix = np.transpose(pd.DataFrame(top_n_together))
        # print(type(top_n_together_matrix), "\n")

        # Continue working with abosulte values.
        abs_top_n_together_matrix = np.abs(top_n_together_matrix)

        percentage = list(range(n_sensors))
        k = 0
        for i in range(n_sensors):
            # print(top_n_features_matrix.shape)
            # print("Number of features from sensor %2.0f is: %3.0f or  %4.2f %%" % (i, len(top_n_features[i]), len(top_n_features[i])/len(sensor_n)*100))
            percentage[i] = len(top_n_features[i])
            k = k + len(top_n_features[i]) / len(self.sensor_indices) * 100
        # print("----------------------------------------------------")
        # print("                                             %4.2f" % (k))

        return abs_top_n_together_matrix

    def plot_feature_percentages(self, sensor_percentage, labels=None, figsize=(8, 8)):
        """
        Plot pie chart which shows the percentages of features from each sensor

        """
        fig1, ax1 = plt.subplots(figsize=figsize)
        ax1.set_title("Percentages of features from each sensor")
        ax1.pie(sensor_percentage, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, )
        ax1.axis('equal')

        return fig1
