import toml
import dao
import numpy as np
import ctypes
import os
daoLogLevel = ctypes.c_int.in_dll(dao.daoLib, "daoLogLevel")
daoLogLevel.value=0
this_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_script_dir,'../config/shm_path.toml'), 'r') as f:
    shm_path = toml.load(f)

K_mat_flag_shm = dao.shm(shm_path['event_flag']['K_mat_flag'])

K_mat_flag_shm.set_data(np.ones((1,1),dtype = np.uint32))


cl_flag_shm = dao.shm(shm_path['settings']['closed_loop_state_flag'])

cl_flag_shm.set_data(np.ones((1,1),dtype = np.uint32))

pyramid_flag_shm = dao.shm(shm_path['event_flag']['pyramid_flag'])

pyramid_flag_shm.set_data(np.ones((1,1),dtype = np.uint32))