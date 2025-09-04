import toml
import dao
import numpy as np
import ctypes
import os
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0

this_script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(this_script_dir,'../config/config.toml'), 'r') as f:
    config = toml.load(f)
with open(os.path.join(this_script_dir,'../config/shm_path.toml'), 'r') as f:
    shm_path = toml.load(f)

# load parameters
n_fft_display = config['visualizer']['n_fft']
buf_size = config['visualizer']['buf_size']
max_order = config['optimizer']['max_order']
n_modes = config['common']['n_modes']
gain = config['integrator']['gain']
dm_shm = dao.shm(shm_path['HW']['dm'])

# create shm
# time domain data
modes_in_buf = np.zeros((buf_size,n_modes),np.float32)
telemetry = np.zeros((2,n_modes),np.float32)
telemetry_ts = np.zeros((2,1),np.float64)
modes_out_buf = np.zeros((buf_size,n_modes),np.float32)
pol_buf = np.zeros((buf_size,n_modes),np.float32)
# state_mat = np.zeros((2*max_order+1, n_modes),np.float32)
K_mat = np.zeros((2*max_order+1, n_modes),np.float32)
K_mat[0,:] = gain
K_mat[max_order+1, :] = 0.99

# frequency domain data
modes_in_fft = np.ones((int(n_fft_display/2),n_modes),np.float32)
modes_out_fft = np.ones((int(n_fft_display/2),n_modes),np.float32)
pol_fft = np.ones((int(n_fft_display/2),n_modes),np.float32)
f = np.ones((int(n_fft_display/2),1),np.float32)
t = np.zeros((buf_size,1),dtype = np.float32)

# loop variables
delay = np.array([[config['common']['delay']]],dtype = np.float32)
fs = np.array([[config['common']['fs']]],dtype = np.float32)
latency = np.zeros((1,1),dtype = np.float32)


M2V = dao.shm(shm_path['KL_mat']['M2V']).get_data(check=False)
V2M = np.linalg.pinv(M2V)

n_modes_dd = np.array([[n_modes]],dtype = np.uint32)
n_modes_controlled = np.array([[n_modes]],dtype = np.uint32)

record_time = np.zeros((1,1),dtype = np.float32)

uint32_0 = np.zeros((1,1),dtype = np.uint32)


modes = np.zeros((n_modes,1),dtype=np.float32)

dd_update_rate = np.array([[np.inf]],dtype = np.float32)

gain_margin = np.array([[1.2]],dtype = np.float32)
wait_time = np.array([[config['calibration']['wait_time']]],dtype = np.float32)
n_fft_optimizer = np.array([[config['optimizer']['n_fft']]],dtype = np.uint32)
dd_order = np.array([[20]],dtype = np.uint32)

slopes_shm = dao.shm(shm_path['HW']['slopes_4sided'])
slopes = slopes_shm.get_data()
n_slopes = slopes.shape[0]
S2M =np.zeros((n_modes,n_slopes),dtype=np.float32)

n_fft_max = config['optimizer']['n_fft_max']

S_dd = np.ones((n_fft_max,n_modes),dtype=np.float32)
S_omgi = np.ones((n_fft_max,n_modes),dtype=np.float32)
S_int = np.ones((n_fft_max,1),dtype=np.float32)
f_opti = np.ones((n_fft_max,1),dtype=np.float32)
f_opti[:n_fft_optimizer[0][0]]= np.linspace(0.1,fs[0][0]/2,n_fft_optimizer[0][0])[:,np.newaxis]

n_act = dm_shm.get_data().shape[0]
flat = np.zeros((n_act,1),dtype = np.float32)

dao.shm(shm_path['time_domain_buff']['modes_in_buf'],modes_in_buf)
dao.shm(shm_path['time_domain_buff']['modes_out_buf'],modes_out_buf)
dao.shm(shm_path['time_domain_buff']['pol_buf'],pol_buf)
dao.shm(shm_path['time_domain_buff']['t'],t)

# dao.shm(shm_path['K']['state_mat'],state_mat)
dao.shm(shm_path['K']['K_mat_int'],K_mat)
dao.shm(shm_path['K']['K_mat_dd'],K_mat)
dao.shm(shm_path['K']['K_mat_omgi'],K_mat)

dao.shm(shm_path['frequency_domain_buff']['modes_in_fft'],modes_in_fft)
dao.shm(shm_path['frequency_domain_buff']['modes_out_fft'],modes_out_fft)
dao.shm(shm_path['frequency_domain_buff']['pol_fft'],pol_fft)
dao.shm(shm_path['frequency_domain_buff']['f'],f)

dao.shm(shm_path['G']['delay'],delay)
dao.shm(shm_path['G']['fs'],fs)
dao.shm(shm_path['G']['latency'],latency)

dao.shm(shm_path['settings']['closed_loop_state_flag'],uint32_0)
dao.shm(shm_path['settings']['n_modes_dd'],n_modes_dd)
dao.shm(shm_path['settings']['n_modes_controlled'],n_modes_controlled)

dao.shm(shm_path['settings']['dd_update_rate'],dd_update_rate)
dao.shm(shm_path['settings']['dd_order'],dd_order)
dao.shm(shm_path['settings']['controller_select'],uint32_0)
dao.shm(shm_path['settings']['pyramid_select'],uint32_0)
dao.shm(shm_path['settings']['gain_margin'],gain_margin)
dao.shm(shm_path['settings']['wait_time'],wait_time)
dao.shm(shm_path['settings']['n_fft'],n_fft_optimizer)
dao.shm(shm_path['settings']['record_time'],record_time)

dao.shm(shm_path['KL_mat']['S2M'],S2M)
dao.shm(shm_path['KL_mat']['V2M'],V2M)

dao.shm(shm_path['HW']['modes_in_custom'],modes)

dao.shm(shm_path['S']['S_dd'],S_dd)
dao.shm(shm_path['S']['S_omgi'],S_omgi)
dao.shm(shm_path['S']['S_int'],S_int)
dao.shm(shm_path['S']['f_opti'],f_opti)

dao.shm(shm_path['telemetry']['telemetry'],telemetry)
dao.shm(shm_path['telemetry']['telemetry_ts'],telemetry_ts)

dao.shm(shm_path['event_flag']['reset_flag'],uint32_0)
dao.shm(shm_path['event_flag']['K_mat_flag'],uint32_0)
dao.shm(shm_path['event_flag']['pyramid_flag'],uint32_0)