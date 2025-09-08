import dao
import numpy as np
import toml
import time 
import os
import ctypes
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0

this_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_script_dir,'../config/config.toml'), 'r') as f:
    config = toml.load(f)
with open(os.path.join(this_script_dir,'../config/shm_path.toml'), 'r') as f:
    shm_path = toml.load(f)

sem_nb = config['sem_nb']['pol']


delay = dao.shm(shm_path['G']['delay']).get_data()[0][0]
# delay = 2

update_rate = config['visualizer']['update_rate']


telemetry_shm = dao.shm(shm_path['telemetry']['telemetry'])

modes_in_buf_shm = dao.shm(shm_path['time_domain_buff']['modes_in_buf'])
modes_out_buf_shm = dao.shm(shm_path['time_domain_buff']['modes_out_buf'])
pol_buf_shm = dao.shm(shm_path['time_domain_buff']['pol_buf'])


def pol_reconstruct(command_buff, measurement_buff, delay):
    delay_floor = int(np.floor(delay))
    delay_ceil = int(np.ceil(delay))
    delay_frac,_ = np.modf(delay)
    if delay_ceil == delay_floor:
        pol = measurement_buff[-1,:] + command_buff[-delay_ceil-1,:]
    else:
        pol = measurement_buff[-1, :] + (1 - delay_frac) * command_buff[-delay_floor-1, :] + delay_frac * command_buff[-delay_ceil-1,:]
    return pol

pol_buf = pol_buf_shm.get_data(semNb=sem_nb)
modes_in_buf = modes_in_buf_shm.get_data(semNb=sem_nb)
modes_out_buf = modes_out_buf_shm.get_data(semNb=sem_nb)

time_start = time.perf_counter()

while True:
    new_time = time.time()
    # print(1/(new_time-old_time))
    old_time = new_time
    telemetry = telemetry_shm.get_data(check=True, semNb=sem_nb)
    modes_in = telemetry[0, :]
    command =  telemetry[1, :]

    pol_buf = np.roll(pol_buf, -1, axis=0)
    modes_in_buf = np.roll(modes_in_buf, -1, axis=0)
    modes_out_buf = np.roll(modes_out_buf, -1, axis=0)

    modes_in_buf[-1, :] = modes_in
    modes_out_buf[-1, :] = command
    pol = pol_reconstruct(modes_out_buf, modes_in_buf, delay)
    pol_buf[-1, :] = pol

    if (time.perf_counter() - time_start > update_rate):
        modes_in_buf_shm.set_data(modes_in_buf.astype(np.float32))
        modes_out_buf_shm.set_data(modes_out_buf.astype(np.float32))
        pol_buf_shm.set_data(pol_buf.astype(np.float32))
        time_start = time.perf_counter()