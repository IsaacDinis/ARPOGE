import toml
import numpy as np
from dd_utils import *
import dd4ao
import time
import sys
import contextlib
import os
import dao 
import ctypes
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0

this_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_script_dir,'../config/config.toml'), 'r') as f:
    config = toml.load(f)
with open(os.path.join(this_script_dir,'../config/shm_path.toml'), 'r') as f:
    shm_path = toml.load(f)


sem_nb = config['sem_nb']['optimizer']
max_order = config['optimizer']['max_order']

update_rate = config['optimizer']['update_rate']
n_fft = int(dao.shm(shm_path['settings']['n_fft']).get_data(check=False, semNb=sem_nb)[0][0])
fs = dao.shm(shm_path['G']['fs']).get_data(check=False, semNb=sem_nb)[0][0]
delay = dao.shm(shm_path['G']['delay']).get_data(check=False, semNb=sem_nb)[0][0]

K_mat_shm = dao.shm(shm_path['K']['K_mat_dd'])
n_modes = dao.shm(shm_path['settings']['n_modes_dd']).get_data(check=False, semNb=sem_nb)[0][0]
order = dao.shm(shm_path['settings']['dd_order']).get_data(check=False, semNb=sem_nb)[0][0]
gain_margin = dao.shm(shm_path['settings']['gain_margin']).get_data(check=False, semNb=sem_nb)[0][0]
high_freq_weight = dao.shm(shm_path['settings']['high_freq_weight']).get_data(check=False, semNb=sem_nb)[0][0]
f_shm = dao.shm(shm_path['frequency_domain_buff']['f'])
pol_fft_shm = dao.shm(shm_path['frequency_domain_buff']['pol_fft'])

S_shm = dao.shm(shm_path['S']['S_dd']) 
f_opti_shm = dao.shm(shm_path['S']['f_opti'])


print("starting optimization")

S = S_shm.get_data(check=False, semNb=sem_nb)
K_array  = np.empty(n_modes,dtype = dd4ao.DD4AO)

t_start = time.perf_counter()

K_mat = K_mat_shm.get_data(check=False, semNb=sem_nb)

f_p =  f_shm.get_data(check=False, semNb=sem_nb).squeeze()
f = np.linspace(f_p[0],f_p[-1],n_fft)

f_opti_shm.set_data(f[:,np.newaxis].astype(np.float32)) 
w = 2*np.pi*f

G_resp = G_freq_resp(delay, w, fs)*gain_margin

# G_resp = freqresp(G_tf(delay,fs),w)*gain_margin
start = 7
powers = powers_of_two_between(start,n_modes)
optimization_indexes = np.concatenate((np.arange(start),powers))
n_optmization = optimization_indexes.shape[0]

pol_fft_p = pol_fft_shm.get_data(check=False, semNb=sem_nb) 
pol_fft = np.zeros((n_fft,n_modes))  
for i in range(n_modes):
    pol_fft[:,i] = np.interp(f,f_p,pol_fft_p[:,i])

optimization_indexes_wide = np.searchsorted(optimization_indexes, np.arange(n_modes), side='right') - 1
pol_fft_avg = np.zeros((pol_fft.shape[0], n_optmization))

# Vectorized accumulation (like a histogram with weights)
np.add.at(pol_fft_avg, (slice(None), optimization_indexes_wide), pol_fft)

# Compute bin counts: how many modes per optimization bin
bin_counts = list(np.diff(optimization_indexes))
bin_counts.append(n_modes - optimization_indexes[-1])
bin_counts = np.array(bin_counts)

# Normalize each bin
pol_fft_avg /= bin_counts


for i in range(n_optmization):
    print(i)
    K_array[i] = dd4ao.DD4AO(w, G_resp, pol_fft_avg[:,i], order,fs, n_iter = 10, tol = 1e-2,high_freq_u_lim=True, high_freq_weight = high_freq_weight)
    K_array[i].compute_controller()
    
for i in range(n_modes):
    K_mat[:order + 1,i] = K_array[optimization_indexes_wide[i]].num.squeeze()
    K_mat[max_order + 1:max_order+order+1,i] = -K_array[optimization_indexes_wide[i]].den.squeeze()[1:]
    K_mat_shm.set_data(K_mat)
    S[:n_fft,i] = 1/np.abs(K_array[optimization_indexes_wide[i]].S_freq.squeeze())

S_shm.set_data(S)
elapsed_time = time.perf_counter() - t_start
print('Controller optimized in = {:.2f} s'.format(elapsed_time))


# time.sleep(30)


# G = G_tf(delay, fs)
# for i in range(n_modes):
#     if(check_K_stability(K_array[i].K,G)>1):
#         print(i)
#         print("unstable")
#         stop

# mode = 6

# K = K_array[mode].K
# fig, axs = plt.subplots(1, 1, figsize=(8, 12))  # 3 subplots stacked vertically
# plot_comp_sensitivity(axs, G, K, K, f)


# fig, axs = plt.subplots(1, 1, figsize=(8, 12))  # 3 subplots stacked vertically
# plot_sensitivity(axs, G, K, K, np.interp(f,f_p,pol_fft_p[:,mode]),f,10)
# # plt.show()

# mode = 1 
# K = K_array[mode].K
# fig, axs = plt.subplots(1, 1, figsize=(8, 12))  # 3 subplots stacked vertically
# plot_comp_sensitivity(axs, G, K, K, f)


# fig, axs = plt.subplots(1, 1, figsize=(8, 12))  # 3 subplots stacked vertically
# plot_sensitivity(axs, G, K, K, np.interp(f,f_p,pol_fft_p[:,mode]),f,10)
# plt.show()

# print(check_K_stability(K_array[mode].K,G))
