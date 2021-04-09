import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Matrix, sin, cos
from sympy import init_printing

init_printing(use_latex=True)

# enter lossweight
uweight = 4
pweight = 5.5

numstates = 6  # States
dt = 0.1 / 1.0  # Sample Rate of the Measurements is 1Hz

vs, psis, dpsis, dts, xs, ys, lats, lons, axs = symbols('v \psi \dot\psi T x y lat lon a')

gs = Matrix([[xs + (vs / dpsis) * (sin(psis + dpsis * dts) - sin(psis))],
             [ys + (vs / dpsis) * (-cos(psis + dpsis * dts) + cos(psis))],
             [psis + dpsis * dts],
             [axs * dts + vs],
             [dpsis],
             [axs]])
state = Matrix([xs, ys, psis, vs, dpsis, axs])

P = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
sGPS = 0.5 * 8.8 * dt ** 2  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sCourse = 0.1 * dt  # assume 0.1rad/s as maximum turn rate for the vehicle
sVelocity = 8.8 * dt  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sYaw = 1.0 * dt  # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle
sAccel = 0.5

Q = np.diag([sGPS ** 2, sGPS ** 2, sCourse ** 2, sVelocity ** 2, sYaw ** 2, sAccel ** 2])

# Load GPS data
datafile0 = './Updated/' + str(uweight) + ' ' + str(pweight) + 'sanitized_gps.txt'
# datafile0 = "C:/Users/Jonathan Seet/PycharmProjects/tensorenv/Plot/PROCESSED DATA/motor_RAW_GPS_NORMAL.txt"
Timestamp, \
Speed, \
Latitude, \
Longitude, \
Altitude, \
Vertical_accuracy, \
Horizontal_accuracy, \
Course, \
Difcourse, \
non0, \
non1, \
non2 = np.loadtxt(datafile0, unpack=True)
# A course of 0° means the Car is traveling north bound
# and 90° means it is traveling east bound.
# In the Calculation following, East is Zero and North is 90°
# We need an offset.
Course = (-Course + 90.0)
# load Accelerometer data
datafile1 = './Updated/' + str(uweight) + ' ' + str(pweight) + 'sanitized_acc.txt'
# datafile1='C:/Users/Jonathan Seet/PycharmProjects/tensorenv/Plot/PROCESSED DATA/motor_RAW_ACCELEROMETERS_NORMAL.txt'

Timestamp1, \
bs, \
ax1, \
ay, \
az, \
ax_kf, \
ay_kf, \
ax, \
roll, \
pitch, \
yaw = np.loadtxt(datafile1, unpack=True)
# ax Gs
ax = ax * 10
# calculate yawrate
yawold = yaw
yaw = np.array(yaw)
yaw = np.delete(yaw, 0, axis=0)
yaw = np.append(yaw, yawold[-1])
yawrate = yaw - yawold

hs = Matrix([[psis],
             [vs],
             [dpsis],
             [axs]])
JHs = hs.jacobian(state)

varc = 0.1
varspeed = 5.0  # Variance of the speed measurement
varyaw = 0.1  # Variance of the yawrate measurement
varacc = 1.0  # Variance of the longitudinal Acceleration
R = np.diag([varc ** 2, varspeed ** 2, varyaw ** 2, varacc ** 2])

I = np.eye(numstates)
# Approx. Lat/Lon to Meters to check Location
RadiusEarth = 6378388.0  # m
arc = 2.0 * np.pi * (RadiusEarth + Altitude) / 360.0  # m/°
dx = arc * np.cos(Latitude * np.pi / 180.0) * np.hstack((0.0, np.diff(Longitude)))  # in m
dy = arc * np.hstack((0.0, np.diff(Latitude)))  # in m
mx = np.cumsum(dx)
my = np.cumsum(dy)
ds = np.sqrt(dx ** 2 + dy ** 2)
GPS = (ds != 0.0).astype('bool')  # GPS Trigger for Kalman Filter

# initial state
x = np.matrix([[mx[0], my[0], Course[0] / 180.0 * np.pi, Speed[0] / 3.6 + 0.001, yawrate[0] / 180.0 * np.pi, ax[0]]]).T
measurements = np.vstack((Course / 180.0 * np.pi, Speed / 3.6, yawrate / 180.0 * np.pi, ax))
# Length of the measurement
m = measurements.shape[1]

# Pre-allocation for Plotting
x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
Zx = []
Zy = []
Px = []
Py = []
Pdx = []
Pdy = []
Pddx = []
Pddy = []
Pdv = []
Kx = []
Ky = []
Kdx = []
Kdy = []
Kddx = []
Kdv = []
dstate = []

for filterstep in range(m):
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    # see "Dynamic Matrix"
    x[0] = x[0] + x[3] * dt * np.cos(x[2])
    x[1] = x[1] + x[3] * dt * np.sin(x[2])
    x[2] = x[2]
    x[3] = x[3] + x[5] * dt
    x[4] = 0.0000001  # avoid numerical issues in Jacobians
    x[5] = x[5]
    dstate.append(0)
    # Calculate the Jacobian of the Dynamic Matrix A
    # see "Calculate the Jacobian of the Dynamic Matrix with respect to the state vector"
    a13 = float((x[3] / x[4]) * (np.cos(x[4] * dt + x[2]) - np.cos(x[2])))
    a14 = float((1.0 / x[4]) * (np.sin(x[4] * dt + x[2]) - np.sin(x[2])))
    a15 = float(
        (dt * x[3] / x[4]) * np.cos(x[4] * dt + x[2]) - (x[3] / x[4] ** 2) * (np.sin(x[4] * dt + x[2]) - np.sin(x[2])))
    a23 = float((x[3] / x[4]) * (np.sin(x[4] * dt + x[2]) - np.sin(x[2])))
    a24 = float((1.0 / x[4]) * (-np.cos(x[4] * dt + x[2]) + np.cos(x[2])))
    a25 = float(
        (dt * x[3] / x[4]) * np.sin(x[4] * dt + x[2]) - (x[3] / x[4] ** 2) * (-np.cos(x[4] * dt + x[2]) + np.cos(x[2])))
    JA = np.matrix([[1.0, 0.0, a13, a14, a15, 0.0],
                    [0.0, 1.0, a23, a24, a25, 0.0],
                    [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    # Project the error covariance ahead
    P = JA * P * JA.T + Q

    # Measurement Update (Correction)
    # ===============================
    # Measurement Function
    hx = np.matrix([[float(x[2])],
                    [float(x[3])],
                    [float(x[4])],
                    [float(x[5])]])
    JH = np.matrix([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    S = JH * P * JH.T + R
    K = (P * JH.T) * np.linalg.inv(S)

    # Update the estimate via
    Z = measurements[:, filterstep].reshape(JH.shape[0], 1)
    y = Z - (hx)  # Innovation or Residual
    x = x + (K * y)

    # Update the error covariance
    P = (I - (K * JH)) * P

    # Save states for Plotting
    x0.append(float(x[0]))
    x1.append(float(x[1]))
    x2.append(float(x[2]))
    x3.append(float(x[3]))
    x4.append(float(x[4]))
    x5.append(float(x[5]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))
    Px.append(float(P[0, 0]))
    Py.append(float(P[1, 1]))
    Pdx.append(float(P[2, 2]))
    Pdy.append(float(P[3, 3]))
    Pddx.append(float(P[4, 4]))
    Pdv.append(float(P[5, 5]))
    Kx.append(float(K[0, 0]))
    Ky.append(float(K[1, 0]))
    Kdx.append(float(K[2, 0]))
    Kdy.append(float(K[3, 0]))
    Kddx.append(float(K[4, 0]))
    Kdv.append(float(K[5, 0]))

x0 = np.insert(x0, 0, 0)
x0 = np.delete(x0, -1)
x1 = np.insert(x1, 0, 0)
x1 = np.delete(x1, -1)

# plot position
fig = plt.figure(figsize=(16, 9))
# EKF State
plt.quiver(x0, x1, np.cos(x2), np.sin(x2), color='#94C600', units='xy', width=0.05, scale=0.5)
plt.plot(x0, x1, label='EKF Position', c='k', lw=5)
# Measurements
plt.scatter(mx[::5], my[::5], s=50, label='GPS Measurements', marker='+')
# Start/Goal
plt.scatter(x0[0], x1[0], s=60, label='Start', c='g')
plt.scatter(x0[-1], x1[-1], s=60, label='Goal', c='r')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Position')
plt.legend(loc='best')
plt.axis('equal')


plt.show()
fig.savefig('./EKF GPS/' + str(uweight) + ' ' + str(pweight) + 'EKF GPS.png')

nmse_x = np.linalg.norm((x0 - mx)) / np.linalg.norm(mx)
nmse_y = np.linalg.norm((x1 - my)) / np.linalg.norm(my)
print(nmse_x, nmse_y)