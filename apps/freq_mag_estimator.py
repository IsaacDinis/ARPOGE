import dao
import numpy as np
import time
from dd_utils import *
import toml
import ctypes
import os
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0

this_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_script_dir,'../config/config.toml'), 'r') as f:
    config = toml.load(f)
with open(os.path.join(this_script_dir,'../config/shm_path.toml'), 'r') as f:
    shm_path = toml.load(f)



sem_nb = config['sem_nb']['fft']
n_fft = config['visualizer']['n_fft']
n_modes = config['common']['n_modes']
update_rate = config['freq_mag_estimator']['update_rate']

modes_in_buf_shm = dao.shm(shm_path['time_domain_buff']['modes_in_buf'])
modes_out_buf_shm = dao.shm(shm_path['time_domain_buff']['modes_out_buf'])
pol_buf_shm = dao.shm(shm_path['time_domain_buff']['pol_buf'])

modes_in_fft_shm = dao.shm(shm_path['frequency_domain_buff']['modes_in_fft'])
modes_out_fft_shm = dao.shm(shm_path['frequency_domain_buff']['modes_out_fft'])
pol_fft_shm = dao.shm(shm_path['frequency_domain_buff']['pol_fft'])
f_shm = dao.shm(shm_path['frequency_domain_buff']['f'])

fs = dao.shm(shm_path['G']['fs']).get_data()[0][0]

closed_loop_state_flag_shm = dao.shm(shm_path['settings']['closed_loop_state_flag'])

time_start = time.perf_counter()

while True:
    # if (time.perf_counter() - time_start > update_rate and closed_loop_state_flag_shm.get_data(check=False, semNb=sem_nb)):
    if (time.perf_counter() - time_start > update_rate):
        pol_buf = pol_buf_shm.get_data(check = True, semNb=sem_nb)
        res_buf = modes_in_buf_shm.get_data(check = True, semNb=sem_nb)
        modes_out_buf = modes_out_buf_shm.get_data(check = True, semNb=sem_nb)
        modes_out_buf -= np.mean(modes_out_buf,axis = 0)
        pol_buf -= np.mean(pol_buf,axis = 0)

        # if (pol_buf.any() and res_buf.any() and modes_out_buf.any()):

        pol_fft, f, _ = compute_fft_mag_welch(pol_buf, n_fft, fs)
        res_fft, f, _ = compute_fft_mag_welch(res_buf, n_fft, fs)
        modes_out_fft, f, _ = compute_fft_mag_welch(modes_out_buf, n_fft, fs)

        modes_out_fft = np.clip(modes_out_fft, 1e-5, None)
        res_fft = np.clip(res_fft, 1e-5, None)
        pol_fft = np.clip(pol_fft, 1e-5, None)

        if n_modes == 1:
            pol_fft = pol_fft[:,np.newaxis]
            res_fft = res_fft[:,np.newaxis]
            modes_out_fft = modes_out_fft[:,np.newaxis]
        f = f[:,np.newaxis]
        pol_fft_shm.set_data((pol_fft[1:,:]).astype(np.float32))
        modes_in_fft_shm.set_data((res_fft[1:,:]).astype(np.float32))
        modes_out_fft_shm.set_data((modes_out_fft[1:,:]).astype(np.float32))
        f_shm.set_data(f[1:,:].astype(np.float32))

        time_start = time.perf_counter()
