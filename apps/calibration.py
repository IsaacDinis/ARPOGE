import dao
import numpy as np
import toml
import time 
from matplotlib import pyplot as plt
import os
import ctypes
import astropy.io.fits as fits
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0

this_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_script_dir,'../config/config.toml'), 'r') as f:
    config = toml.load(f)
with open(os.path.join(this_script_dir,'../config/shm_path.toml'), 'r') as f:
    shm_path = toml.load(f)


def get_pixels(pixels_shm,bias):
    return np.clip(pixels_shm.get_data().astype(np.int32) - bias.astype(np.int32), 0, None).astype(np.uint16)

def get_pixels_masked(pixels_shm,bias,mask):
    pixels = get_pixels(pixels_shm,bias)
    return pixels[mask]

n_modes = config['common']['n_modes']
amp = config['calibration']['amp']
sem_nb = config['sem_nb']['calib']
wait_time = config['calibration']['wait_time']
threshold = config['calibration']['threshold']
max_voltage = config['hrtc']['max_voltage']
pixels_shm = dao.shm(shm_path['HW']['pixels_4sided'])

M2V = dao.shm(shm_path['KL_mat']['M2V']).get_data()

pixels = pixels_shm.get_data()
n_pixels = pixels.shape[0]
dm_shm = dao.shm(shm_path['HW']['flat_dm'])

S2M_shm = dao.shm(shm_path['KL_mat']['S2M'])
# M2S = np.zeros((n_pixels,n_modes))

arr_bias = np.zeros_like(pixels,dtype=np.uint32)
arr_mask = np.zeros_like(pixels,dtype=np.uint32)
ref = np.zeros_like(pixels,dtype=np.uint16)
n_voltages = dm_shm.get_data().shape[0]

n = 8
dm_shm.set_data(np.zeros(n_voltages,dtype = np.float32))
for i in range(2**n):
    time.sleep(wait_time)
    arr_bias += pixels_shm.get_data().astype(np.uint32)
bias = (arr_bias >> n).astype(np.uint16)


plt.figure()
plt.imshow(bias)
plt.show()

n = 11
for i in range(2**n):
    voltages = np.random.rand(n_voltages)*max_voltage
    dm_shm.set_data(voltages.astype(np.float32))
    time.sleep(wait_time)
    pixels = get_pixels(pixels_shm,bias)
    arr_mask += pixels
arr_mask = (arr_mask >> n).astype(np.uint16)

plt.figure()
plt.imshow(arr_mask)

threshold = 0.9
mask = (arr_mask > threshold*np.mean(arr_mask))

plt.figure()
plt.imshow(mask)
plt.show()

M2S = np.zeros((np.sum(mask),n_modes))


for i in range(n_modes):
    dm_shm.set_data((M2V[:,i]*amp).astype(np.float32))
    time.sleep(wait_time)
    pixels = get_pixels_masked(pixels_shm,bias,mask)/amp
    M2S[:,i] += pixels.squeeze().copy()/2
    dm_shm.set_data((-M2V[:,i]*amp).astype(np.float32))
    time.sleep(wait_time)
    pixels = get_pixels_masked(pixels_shm,bias,mask)/amp
    M2S[:,i] -= pixels.squeeze().copy()/2


S2M = np.linalg.pinv(M2S)

n_points = 21
start = -amp*2
end = amp*2
x = np.linspace(start, end, num=n_points)
y0 = np.zeros(n_points)

mode = 50

for i in range(n_points):
    dm_shm.set_data((M2V[:,mode]*x[i]).astype(np.float32))
    time.sleep(wait_time)
    pixels = get_pixels_masked(pixels_shm,bias,mask)/amp
    y0[i] = (S2M@pixels)[mode]

plt.figure()
plt.plot(x,y0)
plt.xlabel("amplitude applied")
plt.ylabel("amplitude read")
plt.title(f"mode {mode} calibration amp = {amp:.1f}")

dy0_dx = np.gradient(y0, x)

plt.figure()
plt.plot(x,dy0_dx)
plt.xlabel("amplitude applied")
plt.ylabel("derivative amplitude read")
plt.title(f"mode {mode} calibration amp = {amp:.1f}")

plt.show()
# n_points = 11
# start = -amp/10
# end = amp/10
# x = np.linspace(start, end, num=n_points)
# y = np.zeros((n_points,n_modes_calib))

# for mode in range(n_modes_calib):
#     for i in range(n_points):
#         dm_shm.set_data((M2V[:,mode]*x[i]).astype(np.float32))
#         time.sleep(wait_time)
#         slopes = slopes_shm.get_data().squeeze()
#         y[i,mode] = (S2M@slopes)[mode]

# dy_dx = np.gradient(y, x, axis = 0)
# system_gain = dy_dx[int(n_points/2)+1,:]


# dm_shm.set_data((M2V[:,0]*0).astype(np.float32))


# S2M = S2M/system_gain[:, np.newaxis]

# S2M_shm.set_data(S2M[:n_modes,:].astype(np.float32))

# plt.figure()
# plt.plot(system_gain)
# plt.xlabel("mode")
# plt.ylabel("gain")
# plt.title("system gain")

# plt.show()
fits.writeto("../data/mask.fits",mask.astype(np.uint16),overwrite = True)
fits.writeto("../data/bias.fits",bias,overwrite = True)
fits.writeto("../data/M2S.fits",M2S,overwrite = True)
fits.writeto("../data/S2M.fits",S2M,overwrite = True)
fits.writeto("../data/ref.fits",ref,overwrite = True)

plt.figure()
pixels = pixels_shm.get_data()
plt.imshow(pixels)

plt.show()

plt.figure()
pixels = pixels_shm.get_data()-bias.astype(np.uint16)
plt.imshow(pixels)

plt.show()