import dao
import numpy as np
import toml
from toml_file_updater import TomlFileUpdater
import os
from datetime import datetime
import time 
import astropy.io.fits as fits
import ctypes
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0
# list to record
# slopes
# modes
# command
# M2V
# M2S
# selected mode
# dd order 
# dd n fft
# powers of 2
# n_modes low
# n_modes high
# n_modes controlled 

 
print('starting record')
this_script_dir = os.path.dirname(os.path.abspath(__file__))

folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
record_dir = "../record/"
full_path = os.path.join(this_script_dir,record_dir,folder_name)
os.makedirs(full_path, exist_ok=True)


with open(os.path.join(this_script_dir,'../config/config.toml'), 'r') as f:
    config = toml.load(f)
with open(os.path.join(this_script_dir,'../config/shm_path.toml'), 'r') as f:
    shm_path = toml.load(f)

sem_nb = config['sem_nb']['rec']
n_modes = config['common']['n_modes']
n_voltages = config['common']['n_voltages']
fs = dao.shm(shm_path['G']['fs']).get_data()[0][0]
record_time = dao.shm(shm_path['settings']['record_time']).get_data()[0][0]
# cblue_shm = dao.shm(shm_path['control']['cblue'])
dm_shm = dao.shm(shm_path['HW']['dm'])
dm0_shm = dao.shm(shm_path['HW']['dm0'])
dm1_shm = dao.shm(shm_path['HW']['dm1'])
dm2_shm = dao.shm(shm_path['HW']['dm2'])
dm3_shm = dao.shm(shm_path['HW']['dm3'])
save_slopes_state_flag = dao.shm(shm_path['settings']['save_slopes_state_flag']).get_data()[0][0] 
slopes_shm = dao.shm(shm_path['HW']['slopes_3'])
telemetry_shm = dao.shm(shm_path['telemetry']['telemetry'])
telemetry_ts_shm = dao.shm(shm_path['telemetry']['telemetry_ts']) 
pyramid_select_shm = dao.shm(shm_path['settings']['pyramid_select']) 
flux_shm = dao.shm(shm_path['HW']['flux']) 
epoch = np.datetime64('1970-01-01T00:00:00', 'us')
n_fft = dao.shm(shm_path['settings']['n_fft']).get_data()[0][0]
controller_select = dao.shm(shm_path['settings']['controller_select']).get_data()[0][0]
gain_margin = dao.shm(shm_path['settings']['gain_margin']).get_data()[0][0]
dd_order = dao.shm(shm_path['settings']['dd_order']).get_data()[0][0]

dd_update_rate = dao.shm(shm_path['settings']['dd_update_rate']).get_data()[0][0]
n_modes_controlled = dao.shm(shm_path['settings']['n_modes_controlled']).get_data()[0][0]
n_modes_dd = dao.shm(shm_path['settings']['n_modes_dd']).get_data()[0][0]
record_time = dao.shm(shm_path['settings']['record_time']).get_data()[0][0]
delay = dao.shm(shm_path['G']['delay']).get_data()[0][0]
M2V = dao.shm(shm_path['KL_mat']['M2V']).get_data()
S2M = dao.shm(shm_path['KL_mat']['S2M']).get_data()
int_gain = dao.shm(shm_path['K']['K_mat_int']).get_data()[0,0]

# norm_flux_pyr_img_shm = dao.shm(shm_path_flux['norm_flux_pyr_img'])
# strehl_ratio_shm = dao.shm(shm_path_flux['strehl_ratio'])

record_its = int(record_time*fs)

results_file = TomlFileUpdater(os.path.join(full_path, "results.toml"))
results_file.add('n_fft',n_fft)

match controller_select:
    case 0:
        results_file.add('control mode','int')
        results_file.add('gain',int_gain)
    case 1:
        results_file.add('control mode','dd')
        results_file.add('n_fft',n_fft)
        results_file.add('gain_margin',gain_margin)
        results_file.add('dd_order',dd_order)
        results_file.add('dd_update_rate',dd_update_rate)
        results_file.add('n_modes_dd',n_modes_dd)
    case 2:
        results_file.add('control mode','omgi')
        results_file.add('n_fft',n_fft)
        results_file.add('gain_margin',gain_margin)

match pyramid_select_shm.get_data(check=False)[0][0]:
    case 0:
        results_file.add('pyramid','4 sided')
        modes_in_bis_shm = dao.shm(shm_path['HW']['modes_in_3sided'])
    case 1:
        results_file.add('pyramid','3 sided')
        modes_in_bis_shm = dao.shm(shm_path['HW']['modes_in_4sided'])

results_file.add('delay',delay)
results_file.add('n modes controlled',n_modes_controlled)
results_file.add('record time',record_time)

modes_in_buf = np.zeros((record_its,n_modes))
modes_in_bis_buf = np.zeros((record_its,n_modes))
modes_out_buf = np.zeros((record_its,n_modes))
voltages_buf = np.zeros((record_its,n_voltages))
# pyr_flux_buf = np.zeros((record_its,1))
# strehl_buf = np.zeros((record_its,1))

dm0_buf = np.zeros((record_its,n_voltages))
dm1_buf = np.zeros((record_its,n_voltages))
dm2_buf = np.zeros((record_its,n_voltages))
dm3_buf = np.zeros((record_its,n_voltages))

modes_in_ts_buf = np.zeros((record_its,1),dtype=np.float64)
modes_out_ts_buf = np.zeros((record_its,1),dtype=np.float64)
flux_buf = np.zeros((record_its,1),dtype=np.float64)

if save_slopes_state_flag:
    n_slopes =  slopes_shm.get_data().shape[0]
    slopes_buf = np.zeros((record_its,n_slopes))
# cblue_shape = cblue_shm.get_data(check=False).shape
# cblue_n_frames = 100
# cblue_count = 0
# cblue_n_avg = int(np.ceil(record_its/cblue_n_frames)) 
# cblue_buf = np.zeros((cblue_n_avg,cblue_shape[0],cblue_shape[1]),np.uint32)

for i in range(record_its):

    telemetry = telemetry_shm.get_data(check=True, semNb=sem_nb)
    telemetry_ts = telemetry_ts_shm.get_data(check=False)
    modes_in = telemetry[0, :]
    modes_out =  telemetry[1, :]
    modes_in_ts = telemetry_ts[0, :]
    modes_out_ts =  telemetry_ts[1, :]

    voltages = dm_shm.get_data(check=False).squeeze()
    # pyr_flux = norm_flux_pyr_img_shm.get_data(check=False, semNb=sem_nb).squeeze()
    # strehl = strehl_ratio_shm.get_data(check=False, semNb=sem_nb).squeeze()
    
    dm0 = dm0_shm.get_data(check=False).squeeze()
    dm1 = dm1_shm.get_data(check=False).squeeze()
    dm2 = dm2_shm.get_data(check=False).squeeze()
    dm3 = dm3_shm.get_data(check=False).squeeze()
    
    dm0_buf[i, :] = dm0
    dm1_buf[i, :] = dm1
    dm2_buf[i, :] = dm2
    dm3_buf[i, :] = dm3

    modes_in_buf[i, :] = modes_in
    voltages_buf[i, :] = voltages
    modes_in_ts_buf[i, :] = modes_in_ts 
    modes_out_ts_buf[i, :] =  modes_out_ts 
    modes_out_buf[i, :] = modes_out
    flux_buf[i,:] = flux_shm.get_data(check=False)
    modes_in_bis_buf[i,:] = modes_in_bis_shm.get_data(check=False).squeeze()
    if save_slopes_state_flag:
        slopes_buf[i,:] = slopes_shm.get_data(check=False).squeeze()
    # pyr_flux_buf[i, :] = pyr_flux
    # strehl_buf[i, :] = strehl

    # cblue_buf[cblue_count,:,:] += cblue_shm.get_data(check=False).astype(np.uint32)
    
    # if (i+1) % cblue_n_frames == 0:
    #     cblue_count += 1


# cblue_buf = cblue_buf/cblue_n_avg

rms = np.mean(np.sum(np.square(modes_in_buf),axis=1))
results_file.add('rms',rms)
fits.writeto(os.path.join(full_path, "modes_in_bis.fits"), modes_in_bis_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "modes_in.fits"), modes_in_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "modes_out.fits"), modes_out_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "modes_in_ts.fits"), modes_in_ts_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "modes_out_ts.fits"), modes_out_ts_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "voltages.fits"), voltages_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "pyr_fluxes.fits"), flux_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "M2V.fits"), M2V, overwrite = True)
fits.writeto(os.path.join(full_path, "S2M.fits"), S2M, overwrite = True)

fits.writeto(os.path.join(full_path, "dm0.fits"), dm0_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "dm1.fits"), dm0_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "dm2.fits"), dm0_buf, overwrite = True)
fits.writeto(os.path.join(full_path, "dm3.fits"), dm0_buf, overwrite = True)


if save_slopes_state_flag:
    fits.writeto(os.path.join(full_path, "slopes.fits"), slopes_buf, overwrite = True)
# fits.writeto(os.path.join(full_path, "strehl.fits"), strehl_buf, overwrite = True)
# fits.writeto(os.path.join(full_path, "cblue.fits"), cblue_buf, overwrite = True)
results_file.save()

print('print record done')
