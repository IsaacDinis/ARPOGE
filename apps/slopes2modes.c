#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <sched.h>
#include "dao.h"
#include "toml.h"
#include "utils.h"

// ---------- Structures ----------

struct {
    char *modes;
    char *pixels;
} shm_path;

struct {
    int64_t n_pixels;
} config;

// ---------- Main ----------

int load_shm_path() {
    char errbuf[200];
    toml_table_t *root = load_toml("../../config/shm_path.toml", errbuf, sizeof(errbuf));
    if (!root) return 1;

    toml_table_t *HW = toml_table_in(root, "HW");
    if (HW) {
        toml_rtos(toml_raw_in(HW, "modes_in_custom"), &shm_path.modes);  
        toml_rtos(toml_raw_in(HW, "pixels_3sided"),   &shm_path.pixels);
    }
    toml_free(root);
    return 0;
}

int load_config() {
    char errbuf[200];
    toml_table_t *root = load_toml("../../config/config.toml", errbuf, sizeof(errbuf));
    if (!root) return 1;

    toml_table_t *common    = toml_table_in(root, "common");
    if (common)      toml_rtod(toml_raw_in(common, "n_pixels"),     &config.n_pixels);

    toml_free(root);
    return 0;
}

int main(void) {
  int RT_priority = 93; //any number from 0-99
  struct sched_param schedpar;

  schedpar.sched_priority = RT_priority;
  // r = seteuid(euid_called); //This goes up to maximum privileges
  sched_setscheduler(0, SCHED_FIFO, &schedpar); //other option is SCHED_RR, might be faster
  // r = seteuid(euid_real);//Go back to normal privileges
  load_shm_path();
  load_config();
  long dummy_x, dummy_y; // should be both n_pixels
  long dummy; // should be n_act
  long n_slopes;
  double *bias_image = read_fits_2d("../../calib_data/bias_image.fits", &dummy_x, &dummy_y);
  double *mask = read_fits_2d("../../calib_data/mask.fits", &dummy_x, &dummy_y);
  double *ref_img_norm = read_fits_2d("../../calib_data/reference_image_normalized.fits", &dummy_x, &dummy_y);
  double *IM_KL2S = read_fits_2d("../../calib_data/IM_KL2S.fits", &dummy, &n_slopes);
  printf("n pixels = %ld\n\n", frame_missed);
  printf("n_act = %ld\n\n", frame_missed);
  printf("Frames missed = %ld\n\n", frame_missed);
  return 0;
}
