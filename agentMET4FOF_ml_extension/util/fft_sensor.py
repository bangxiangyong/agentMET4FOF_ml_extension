import numpy as np

class FFT_Sensor():
    """
    Class for specifically handling FFT of sensor data in batches

    """
    def __init__(self, sampling_f=2000, sensor_axis = 1):
        self.sampling_f = sampling_f
        self.sensor_axis = sensor_axis

    def get_freq_axis(self, n_of_samples=2000):
        x_measurements = range(n_of_samples)  # Number of measurements samples in time period.
        x = np.true_divide(x_measurements, n_of_samples)  # Time values, used  as real time axis.
        freq = np.fft.rfftfreq(x.size,
                               (1 / self.sampling_frequency))  # Frequency axis, can be used for ploting in frequency domain.
        return freq

    def fft_sensor(self, sensor_data):
        fft_amplitudes = np.abs(
            np.fft.rfft(sensor_data, axis = 1))  # Ndarray of amplitudes after fourier transform.
        return fft_amplitudes

    def fft_sensor_batch(self, sensors_data):
        """
        Computes fft on 3D array of (num_cycles, cycle_length, num_sensors) dimensions. Expects last dimension to be number of sensors.
        """
        if self.sensor_axis == -1 or self.sensor_axis == 2:
            fft_sensor_data = np.array(
                [self.fft_sensor(sensors_data[:, :, sensor_i]) for sensor_i in range(sensors_data.shape[self.sensor_axis])])
            fft_sensor_data = np.moveaxis(fft_sensor_data, (0, 1, 2), (2, 0, 1))
        else:
            fft_sensor_data = np.array(
                [self.fft_sensor(sensors_data[:, sensor_i]) for sensor_i in range(sensors_data.shape[self.sensor_axis])])
            fft_sensor_data = np.moveaxis(fft_sensor_data, (0, 1, 2), (1, 0, 2))
        return fft_sensor_data

    def transform(self, x_ts, y_train=None):
        """
        For compatibility with agentMET4FOF_ml_extension
        """
        return self.fft_sensor_batch(x_ts)
