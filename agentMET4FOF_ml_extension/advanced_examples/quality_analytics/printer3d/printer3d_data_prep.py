import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as matplot_img
from skimage.transform import resize
import dill
from sklearn.preprocessing import MinMaxScaler

from agentMET4FOF_ml_extension.util.calc_outlier import get_num_outliers_df
from agentMET4FOF_ml_extension.util.fft_sensor import FFT_Sensor
from baetorch.baetorch.util.minmax import MultiMinMaxScaler

output_folder = "E:\\3DPrinter-Dataset\\tables\\Dog bone data.xlsx"

output_pd = pd.read_excel(output_folder)

width_a = output_pd.iloc[1:21,8].values.reshape(-1,1)
width_b = output_pd.iloc[1:21,11].values.reshape(-1,1)
width_samples = np.concatenate((width_a,width_b), axis=1)
width_mean = np.median(width_samples,axis=1)
width_var = np.around(np.abs((width_a-width_b)).astype(float),decimals=2).reshape(-1)

thickness_a = output_pd.iloc[1:21,9].values.reshape(-1,1)
thickness_b = output_pd.iloc[1:21,12].values.reshape(-1,1)
thickness_samples = np.concatenate((width_a,width_b), axis=1)
thickness_mean = np.median(thickness_samples, axis=1)
thickness_var = np.around(np.abs((thickness_a-thickness_b)).astype(float),decimals=2).reshape(-1)


# outputs = np.concatenate((width_mean.reshape(-1,1),thickness_mean.reshape(-1,1), width_var.reshape(-1,1), thickness_var.reshape(-1,1)), axis=1)
outputs = np.concatenate((width_mean.reshape(-1,1),thickness_mean.reshape(-1,1), width_var.reshape(-1,1), thickness_var.reshape(-1,1), width_a, width_b, thickness_a, thickness_b), axis=1)
outputs = np.concatenate((width_a, width_b, thickness_a, thickness_b), axis=1)
# outputs = np.concatenate((width_a, width_b), axis=1)
# outputs = np.concatenate((thickness_a, thickness_b), axis=1)

# Q: are they correlated?..

y_levels = get_num_outliers_df(outputs)

plt.figure()
plt.boxplot(outputs, labels=["Width A","Width B","Thickness A","Thickness B"])

plt.figure()
plt.boxplot(np.concatenate((np.concatenate((width_a,width_b)), np.concatenate((thickness_a,thickness_b))),axis=1),
            labels=["Width","Thickness"])


arg_healthy = np.argwhere(y_levels<1).reshape(-1)
arg_anomaly = np.argwhere(y_levels>=1).reshape(-1)

print("HEALTHY POINTS:"+str(arg_healthy))
print("ANOMALOUS POINTS:"+str(arg_anomaly))

# NOW PREPARE INPUT DATA
main_inp_folder = "E:\\3DPrinter-Dataset\\videos\\"
inp_folder_names = [main_inp_folder+folder for folder in output_pd.iloc[1:21,6].values]

# inp_folder_name_i = inp_folder_names[0]
# inp_images_1 = [jpg_file for jpg_file in os.listdir(inp_folder_name_i) if ".jpg" in jpg_file]
# int_inp_images_1 = [int(jpg_file.split('-')[-1].split("_")[0]) for jpg_file in inp_images_1]
# argsort_inp_images_1 = np.argsort(int_inp_images_1)
# inp_images_1 = np.array(inp_images_1)[argsort_inp_images_1]

# # load and display an image with Matplotlib
def load_image_i(inp_file_name, grayscale=True, resize_factor=10):
    # inp_images_1 = [jpg_file for jpg_file in os.listdir(inp_folder_name_i) if ".jpg" in jpg_file]
    #
    # int_inp_images_1 = [int(jpg_file.split('-')[-1].split("_")[0]) for jpg_file in inp_images_1]
    # argsort_inp_images_1 = np.argsort(int_inp_images_1)
    # inp_images_1 = np.array(inp_images_1)[argsort_inp_images_1]

    # # load and display an image with Matplotlib

    # load image as pixel array
    # image = matplot_img.imread(inp_folder_name_i + "\\" + inp_images_1[0])
    # image = matplot_img.imread(inp_folder_name_i + "\\" + inp_file_name_i)
    image = matplot_img.imread(inp_file_name)

    # summarize shape of the pixel array
    # print(image.dtype)
    # print(image.shape)
    if grayscale:
        image = np.expand_dims((image.mean(-1) / 255.0), -1)
    else:
        image = (image / 255.0)

    # truncate height
    image = image[300:]
    # display the array of pixels as an image

    resize_shape = (int(image.shape[0] / resize_factor),
                    int(image.shape[1] / resize_factor))
    # resize_shape = (100,
    #                 100)
    image_resized = resize(image, resize_shape)
    return image_resized

# plot some images
def plot_image(image_):
    plt.figure()
    plt.imshow(image_,cmap=plt.cm.binary)
    plt.show()

def parse_image_filenames(inp_folder_name_i):
    # inp_folder_name_i = inp_folder_names[0]
    inp_images_1 = [jpg_file for jpg_file in os.listdir(inp_folder_name_i) if ".jpg" in jpg_file]

    int_inp_images_1 = [int(jpg_file.split('-')[-1].split("_")[0]) for jpg_file in inp_images_1]
    argsort_inp_images_1 = np.argsort(int_inp_images_1)
    inp_images_1 = np.array(inp_images_1)[argsort_inp_images_1]
    inp_images_1_ = [inp_folder_name_i+"\\"+filename for filename in inp_images_1]
    return inp_images_1_

def load_video_array(img_folder_name, grayscale=True, resize_factor=10):
    img_filenames = parse_image_filenames(img_folder_name)
    image_resized = np.array([load_image_i(inp_file_name=img_filename, grayscale=grayscale, resize_factor=resize_factor) for img_filename in img_filenames])
    return image_resized


# grayscale= False
# resize_factor = 10
# videos = [load_video_array(inp_folder_name,grayscale=grayscale) for inp_folder_name in inp_folder_names]
#
# plot_image(videos[0][0])
#
# # save parsed file
# parsed_data = {"videos":videos, "targets":y_levels}
# parsed_data_file = "printer3d_parsed2.p"
# if grayscale:
#     parsed_data_file = "printer3d_parsed_" +str(resize_factor)+ "gray.p"
# else:
#     parsed_data_file = "printer3d_parsed_" +str(resize_factor)+ "color.p"
#
# #
# dill.dump(parsed_data, open(parsed_data_file, "wb"))
#

# load hotend and bed readings

df_csv = pd.read_csv(inp_folder_names[1]+"\\"+"print_log.csv")[["hotend","bed"]]

plt.figure()
plt.plot(df_csv)

fig, (ax1,ax2) = plt.subplots(2,1)

for i,inp_folder in enumerate(inp_folder_names):
    df_csv = pd.read_csv(inp_folder + "\\" + "print_log.csv")[["hotend", "bed"]]
    if i in arg_healthy:
        color = "tab:green"
    else:
        color = "tab:red"
    ax1.plot(df_csv["hotend"], color=color)
    ax2.plot(df_csv["bed"], color=color)

num_samples = [len(os.listdir(inp_folder_name))-1 for inp_folder_name in inp_folder_names]
min_samples = np.min(num_samples)

sensors_data = []
for i,inp_folder in enumerate(inp_folder_names):
    df_csv = pd.read_csv(inp_folder + "\\" + "print_log.csv")[["hotend", "bed"]]
    sensors_data.append(df_csv.values[:min_samples-1])
sensors_data= np.array(sensors_data)

fft_sensor_class = FFT_Sensor(sensor_axis=2)

fft_data = fft_sensor_class.fft_sensor_batch(sensors_data)

fig, (ax1,ax2) = plt.subplots(2,1)
for i,data in enumerate(fft_data):
    if i in arg_healthy:
        color = "tab:green"
    elif i in arg_anomaly:
        color = "tab:red"
    ax1.plot(data[:,0], color=color)
    ax2.plot(data[:,1], color=color)

dill.dump({"fft_sensors":fft_data,"targets":y_levels}, open("parsed_fft_data_file.p", "wb"))

# minmax_scaler = MultiMinMaxScaler()
# fft_scaled = minmax_scaler.fit_transform(fft_data[arg_healthy])
#
#
# fig, (ax1,ax2) = plt.subplots(2,1)
# for data in fft_healthy_scaled:
#     color = "tab:green"
#     ax1.plot(data[:,0], color=color)
#     ax2.plot(data[:,1], color=color)
# for data in fft_anomaly_scaled:
#     color = "tab:red"
#     ax1.plot(data[:,0], color=color)
#     ax2.plot(data[:,1], color=color)

# fft_scaled = fft_data.copy()
# for sensor_i in range(fft_data.shape[-1]):
#     for sample_i in range(fft_data.shape[0]):
#         scaler = MinMaxScaler()
#         temp = scaler.fit_transform(fft_data[sample_i,:,sensor_i].reshape(-1,1))
#         fft_scaled[sample_i, :, sensor_i] =temp.reshape(-1)
#
# fig, (ax1,ax2) = plt.subplots(2,1)
# for i,data in enumerate(fft_scaled):
#     if i in arg_healthy:
#         color = "tab:green"
#     elif i in arg_anomaly:
#         color = "tab:red"
#     ax1.plot(data[:,0], color=color)
#     ax2.plot(data[:,1], color=color)







