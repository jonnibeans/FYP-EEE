import numpy as np
from scipy.io import savemat, loadmat

raw_data1 = np.loadtxt("C:/Users/Jonathan Seet/PycharmProjects/tensorenv/Plot/PROCESSED DATA/motor_RAW_GPS_NORMAL.txt")
s1 = raw_data1.shape[0]
raw_data2 = np.loadtxt("C:/Users/Jonathan Seet/PycharmProjects/tensorenv/Plot/PROCESSED DATA/motor_RAW_GPS_DRWOSY.txt")
s2 = raw_data2.shape[0]
raw_data3 = np.loadtxt("C:/Users/Jonathan Seet/PycharmProjects/tensorenv/Plot/PROCESSED DATA/motor_RAW_GPS_AGGRESSIVE.txt")
s3 = raw_data3.shape[0]

gps_raw_data = np.vstack((raw_data1, raw_data2, raw_data3))
label = np.ones((gps_raw_data.shape[0], 1))
label[0:s1] = label[0:s1] * 0
label[s1:s1+s2] = label[s1:s1+s2] * 1
label[s1+s2:s1+s2+s3] = label[s1+s2:s1+s2+s3] * 2

raw_data1 = np.loadtxt("C:/Users/Jonathan Seet/PycharmProjects/tensorenv/Plot/PROCESSED DATA/motor_RAW_ACCELEROMETERS_NORMAL.txt")
raw_data2 = np.loadtxt("C:/Users/Jonathan Seet/PycharmProjects/tensorenv/Plot/PROCESSED DATA/motor_RAW_ACCELEROMETERS_DRWOSY.txt")
raw_data3 = np.loadtxt("C:/Users/Jonathan Seet/PycharmProjects/tensorenv/Plot/PROCESSED DATA/motor_RAW_ACCELEROMETERS_AGGRESSIVE.txt")

acc_raw_data = np.vstack((raw_data1, raw_data2, raw_data3))

""" select part of raw data """
gps_sel = [1, 7]
acc_sel = [5, 6, 7, 8, 9, 10]

x_data = np.hstack((gps_raw_data[:, gps_sel], acc_raw_data[:, acc_sel]))

np.set_printoptions(threshold=np.inf)
file = open("./rawxdata.txt", "w")
file.write(str(x_data))
file.close()

# normalize
x_min = np.min(x_data, axis=0)
x_max = np.max(x_data, axis=0)
x_data = (x_data - x_min) / (x_max-x_min)
print(x_data.shape)

y_pos = gps_raw_data[:, 2:5]

# """ convert gps coordinates """
# coordinate_converter = Transformer.from_crs("epsg:4326", "epsg:2062")
# x_, y_, z_ = coordinate_converter.transform(y_pos[:, 0], y_pos[:, 1], y_pos[:, 2])
# y_pos[:, 0] = x_
# y_pos[:, 1] = y_
# y_pos[:, 2] = z_
#
# plt.plot(y_pos[:, 0], y_pos[:, 1])
# plt.show()

data = []
pos = []
db = []
print(len(x_data))
for i in range(64, len(x_data), 16):
    if label[i-64] == label[i]:
        data.append(x_data[i-64:i])
        pos.append(y_pos[i-64:i])
        db.append(label[i])

x_data = np.array(data)
y_pos = np.array(pos)
y_db = np.array(db)
print(x_data.shape)
print(y_pos.shape)
print(y_db.shape)


savemat('C:/Users/Jonathan Seet/PycharmProjects/tensorenv/Plot/PROCESSED DATA/RawData.mat', {'x_data': x_data, 'y_pos': y_pos, 'y_db': y_db})


