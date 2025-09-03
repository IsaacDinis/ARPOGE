import dao
import numpy as np
import time 
import ctypes
from matplotlib import pyplot as plt
import astropy.io.fits as fits
from datetime import datetime
import os

daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0

verbose = 0
n_modes = 195
sem_nb = 8
order = 20
gain = 0.3
n_iter = 5000
bootstrap_n_iter = 10
print_rate = 1 

modes_shm = dao.shm('/tmp/papyrus_modes.im.shm')
dm_shm = dao.shm('/tmp/dmCmd02.im.shm')
M2V = dao.shm('/tmp/m2c.im.shm').get_data(check=False, semNb=sem_nb)

this_script_dir = os.path.dirname(os.path.abspath(__file__))
folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"_python"
parent_dir = "../results/"
full_path = os.path.join(this_script_dir,parent_dir, folder_name)
os.makedirs(full_path, exist_ok=True)

state_mat = np.zeros((2*order+1, n_modes),np.float32)
K_mat = np.zeros((2*order+1, n_modes),np.float32)
K_mat[0,:] = gain
K_mat[order+1, :] = 0.99

old_time = time.time()
time_at_last_print = time.perf_counter()
last_loop_time = time.perf_counter()
counter = 0
computation_time = 0
write_time = 0
wfs_time = 0
loop_time_array = np.zeros(int(n_iter))
wfs_timestamp_array = np.zeros(int(n_iter))
epoch = np.datetime64('1970-01-01T00:00:00', 'us')

for i in range(n_iter):

    loop_time_array[i] = time.perf_counter() - last_loop_time
    last_loop_time = time.perf_counter()

    start_wfs_time = time.perf_counter()
    modes = modes_shm.get_data(check=True, semNb=sem_nb).squeeze()
    wfs_timestamp_array[i] = (np.datetime64(modes_shm.get_timestamp(), 'us') - epoch) / np.timedelta64(1, 's')
    wfs_time += time.perf_counter() - start_wfs_time


    start_computation_time = time.perf_counter()

    state_mat[1:, :] = state_mat[0:-1, :]
    state_mat[0, :] = modes
    command_mat = np.multiply(state_mat, K_mat)
    command = np.sum(command_mat, axis=0)


    voltage = -M2V[:,:] @ command
    state_mat[order, :] = command

    computation_time += time.perf_counter() - start_computation_time
    start_write_time = time.perf_counter()
    
    dm_shm.set_data(voltage.astype(np.float32))

    write_time += time.perf_counter() - start_write_time

    if verbose:

        if(time.perf_counter() - time_at_last_print > 1):

            loop_time_sub_array = loop_time_array[i-counter:i]
            loop_time_mean = np.sum(loop_time_sub_array)/counter

            print('Mean Loop rate = {:.2f} Hz'.format(1/loop_time_mean))
            print('Mean loop time  = {:.2f} ms'.format(loop_time_mean*1e3))
            print('Mean WFS time  = {:.2f} ms'.format(wfs_time/counter*1e3))
            print('Mean Computation_time = {:.2f} ms'.format(computation_time/counter*1e3))
            print('Mean Write time  = {:.2f} ms'.format(write_time/counter*1e3))
            print('Jitter std  = {:.2f} ms'.format(np.std(loop_time_sub_array)*1e3))
            print('Max loop time  = {:.2f} ms'.format(np.max(loop_time_sub_array*1e3)))
            print('Framed missed  = {:.2f} '.format(np.sum(loop_time_sub_array>2*loop_time_mean)))
            print('\n')
            counter = 0
            computation_time = 0
            write_time = 0
            wfs_time = 0
            time_at_last_print = time.perf_counter()

        counter += 1
    

fits.writeto(os.path.join(full_path, "loop_time_array.fits"), loop_time_array[bootstrap_n_iter:], overwrite = True)
fits.writeto(os.path.join(full_path, "wfs_timestamp_array.fits"), wfs_timestamp_array[bootstrap_n_iter:], overwrite = True)

# plt.figure()
# plt.plot(np.diff(loop_time_array[bootstrap_n_iter:]))

# plt.figure()
# plt.plot(np.diff(wfs_timestamp_array[bootstrap_n_iter:]))

# plt.show()