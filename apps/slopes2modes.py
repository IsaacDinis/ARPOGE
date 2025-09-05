import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy
from matplotlib.colors import LogNorm
import cupy as cp
import toml
import dao
import time
this_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_script_dir,'../config/shm_path.toml'), 'r') as f:
    shm_path = toml.load(f)

# slopes_shm = dao.shm(shm_path['HW']['slopes_3sided'])
modes_shm = dao.shm(shm_path['HW']['modes_in_custom'])
pixels_shm = dao.shm(shm_path['HW']['pixels_3sided'])

calib_data_dir = os.path.join(this_script_dir, "../calib_data/")
bias_image = fits.getdata(os.path.join(calib_data_dir, "bias_image.fits"))
mask = fits.getdata(os.path.join(calib_data_dir, "mask.fits"))
reference_image_normalized = fits.getdata(os.path.join(calib_data_dir, "reference_image_normalized.fits"))
IM_KL2S = fits.getdata(os.path.join(calib_data_dir, "IM_KL2S.fits"))

def compute_pyr_slopes(normalized_pyr_img, normalized_ref_img):
    slopes_image = normalized_pyr_img - normalized_ref_img
    return slopes_image

def bias_correction(image, bias_image):

    image = np.asarray(image)
    bias_image = np.asarray(bias_image)
    corrected_image = image - bias_image
    return corrected_image

def normalize_image(image, mask, bias_img):

    bias_corrected_image = bias_correction(image, bias_img)
    masked_image = bias_corrected_image * mask
    norm_flux = np.abs(np.sum(masked_image))
    normalized_image = masked_image / norm_flux
    # norm_flux_pyr_img_shm.set_data(np.array([[norm_flux]]).astype(np.float32)) # setting shared memory
    return normalized_image


def get_slopes_image(mask, bias_image, reference_image_normalized):

    pyr_img = pixels_shm.get_data(check=True, semNb=5)

    normalized_pyr_img = normalize_image(pyr_img, mask, bias_image)
    slopes_image = compute_pyr_slopes(normalized_pyr_img, reference_image_normalized)
    # print('slopes_image data type', slopes_image.dtype)
    # print('slopes_image shape', slopes_image.shape)
    # slopes_image_shm.set_data(slopes_image.astype(np.float64))
    return slopes_image
    

# Get valid pixel indices from the cropped mask
valid_pixels_indices = np.where(mask > 0)


RM_S2KL = np.linalg.pinv(IM_KL2S, rcond=0.10)

# hdu = fits.PrimaryHDU(RM_S2KL.astype(np.float32).T)
# hdu.writeto(os.path.join(calib_data_dir, "RM_S2KL.fits"),overwrite = True)

print(f"Shape of the reconstruction matrix: {RM_S2KL.shape}")
RM_S2KL_cp = cp.asarray(RM_S2KL)

print('Running AO open loop')

counter = 0

print_rate = 1 # [s]
time_at_last_print = time.perf_counter()
last_loop_time = time.perf_counter()
counter = 0
read_time = 0
computation_time = 0
write_time = 0
loop_time = 0


while True:

    loop_time += time.perf_counter() - last_loop_time
    last_loop_time = time.perf_counter()
    start_read_time = time.perf_counter()

    # Capture and process WFS image
    slopes_image = get_slopes_image(
        mask,
        bias_image,
        reference_image_normalized,
    )

    # Compute slopes
    slopes = slopes_image[valid_pixels_indices].flatten()

    read_time += time.perf_counter() - start_read_time
    start_computation_time = time.perf_counter()
    
    # Compute KL modes present
    slopes_cp = cp.asarray(slopes)
    computed_modes = cp.matmul(slopes_cp, RM_S2KL_cp)
    #computed_modes = slopes @ RM_S2KL

    computation_time += time.perf_counter() - start_computation_time
    start_write_time = time.perf_counter()
    modes_shm.set_data(np.asanyarray(computed_modes.get()).astype(np.float32)) # setting shared memory
    write_time += time.perf_counter() - start_write_time

    counter += 1  # increment loop index

    if(time.perf_counter() - time_at_last_print > print_rate):
        print('Loop rate = {:.2f} Hz'.format(1/(loop_time/counter)))
        print('Loop time  = {:.2f} ms'.format(loop_time/counter*1e3))
        print('Read time = {:.2f} ms'.format(read_time/counter*1e3))
        print('Computation_time = {:.2f} ms'.format(computation_time/counter*1e3))
        print('Write time  = {:.2f} ms'.format(write_time/counter*1e3))
        counter = 0
        read_time = 0
        computation_time = 0
        write_time = 0
        loop_time = 0
        time_at_last_print = time.perf_counter()

