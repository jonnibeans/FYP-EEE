from numpy import loadtxt
import numpy as np
from tensorflow import keras

#updating loss weight file
uweight = 4
pweight = 5.5

datafile0 = "C:/Users/Jonathan Seet/PycharmProjects/tensorenv/Plot/PROCESSED DATA/motor_RAW_GPS_NORMAL.txt"
datafile1 = 'C:/Users/Jonathan Seet/PycharmProjects/tensorenv/Plot/PROCESSED DATA/motor_RAW_ACCELEROMETERS_NORMAL.txt'

gps_raw = loadtxt(datafile0, comments="#", unpack=False)
acc_raw = loadtxt(datafile1, comments="#", unpack=False)

gps_sel = [1, 7]
acc_sel = [5, 6, 7, 8, 9, 10]

# normalizing raw data
x_data = np.hstack((gps_raw[:8512, gps_sel], acc_raw[:8512, acc_sel]))
x_min = np.min(x_data, axis=0)
x_max = np.max(x_data, axis=0)
x_data = (x_data - x_min) / (x_max - x_min)

reshape_x_data = x_data.reshape((133, 64, 8))
print(reshape_x_data.shape)

# loading sanitizer to get sanitized data
sanitizer = keras.models.load_model('./Sanitizers/' + str(uweight) + ' ' + str(pweight) + 'sanitizer.h5')
sanitized_data = sanitizer.predict(reshape_x_data)
reshaped_sanitized_data = sanitized_data.reshape(-1, sanitized_data.shape[-1])

# unpacking sanitized data
reshaped_sanitized_data = (reshaped_sanitized_data + x_min) * (x_max - x_min)


# placing fetched data back into gps and accelerometer data
gps_raw[:8512, gps_sel] = reshaped_sanitized_data[:, 0:2]
acc_raw[:8512, acc_sel] = reshaped_sanitized_data[:, 2:]

print("Saving Updated GPS...")
np.savetxt('./Updated/' + str(uweight) + ' ' + str(pweight) + 'sanitized_gps.txt', gps_raw)
print("... done")
np.savetxt('./Updated/' + str(uweight) + ' ' + str(pweight) + 'sanitized_acc.txt', acc_raw)
print("Saving Updated Accelerometer...")
print("... done")
# print(gps_raw.shape)
# print(acc_raw.shape)
# print(sandata.shape)
