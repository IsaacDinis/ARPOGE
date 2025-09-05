#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <sched.h>
#include <math.h>
#include "dao.h"
#include "toml.h"
#include "utils.h"

// ---------- Structures ----------

struct {
    char *modes;
    char *pixels;
    char *mask;
    char *bias_image;
    char *ref_img_norm;
    char *S2M;
} shm_path;

struct {
    int64_t n_pixels;
    int64_t sem_nb;

} config;

static int end = 0;               // termination flag
// termination function for SIGINT callback
static void endme(){
    end = 1;
}

// ---------- Main ----------

int load_shm_path() {
    char errbuf[200];
    toml_table_t *root = load_toml("../config/shm_path.toml", errbuf, sizeof(errbuf));
    if (!root) return 1;

    toml_table_t *HW = toml_table_in(root, "HW");
    if (HW) {
        toml_rtos(toml_raw_in(HW, "modes_in_custom"), &shm_path.modes);  
        toml_rtos(toml_raw_in(HW, "pixels_3sided"),   &shm_path.pixels);
    }
    toml_table_t *calibration = toml_table_in(root, "calibration");
    if (calibration) {
        toml_rtos(toml_raw_in(calibration, "mask"),         &shm_path.mask);  
        toml_rtos(toml_raw_in(calibration, "bias_image"),   &shm_path.bias_image);
        toml_rtos(toml_raw_in(calibration, "ref_img_norm"), &shm_path.ref_img_norm);
        toml_rtos(toml_raw_in(calibration, "S2M"),          &shm_path.S2M);
    }
    toml_free(root);
    return 0;
}

int load_config() {
    char errbuf[200];
    toml_table_t *root = load_toml("../config/config.toml", errbuf, sizeof(errbuf));
    if (!root) return 1;

    toml_table_t *common    = toml_table_in(root, "common");
    toml_table_t *sem_nb    = toml_table_in(root, "sem_nb");
    if (common) toml_rtoi(toml_raw_in(common, "n_pixels"),  &config.n_pixels);
    if (sem_nb)    toml_rtoi(toml_raw_in(sem_nb, "hrtc"),          &config.sem_nb);

    toml_free(root);
    return 0;
}

int real_time_loop(){
  signal(SIGINT, endme);
  struct timespec timeout;
  IMAGE *pixels_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *modes_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *mask_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *bias_image_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *ref_img_norm_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *S2M_shm = (IMAGE*) malloc(sizeof(IMAGE));
  daoShmShm2Img(shm_path.modes, modes_shm);
  daoShmShm2Img(shm_path.pixels, pixels_shm);
  daoShmShm2Img(shm_path.mask, mask_shm);
  daoShmShm2Img(shm_path.bias_image, bias_image_shm);
  daoShmShm2Img(shm_path.ref_img_norm, ref_img_norm_shm);
  daoShmShm2Img(shm_path.S2M, S2M_shm);

  uint32_t n_pix = pixels_shm->md[0].size[0] ;
  uint32_t n_slopes = S2M_shm->md[0].size[1];
  uint32_t n_modes = S2M_shm->md[0].size[0];
  float* modes  = calloc(n_modes, sizeof(float));
  float* slopes = malloc(n_slopes * sizeof(float));
  float* masked_image = malloc(n_pix * n_pix * sizeof(float));

  float norm_flux = 0.0f;
  while(!end){
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += 1; // 1 second timeout
    if (daoShmWaitForSemaphoreTimeout(pixels_shm, config.sem_nb, &timeout) != -1){
      modes_shm->md[0].cnt2 = pixels_shm->md[0].cnt2;
      norm_flux = 0.0f;
      for (uint32_t i = 0; i < n_pix; i++) {
        for (uint32_t j = 0; j < n_pix; j++) {
          uint32_t idx = i * n_pix + j;
          float corrected = (float)pixels_shm->array.UI16[idx] - (float)bias_image_shm->array.UI16[idx];
          float masked = corrected * (float)mask_shm->array.UI16[idx];
          masked_image[idx] = masked;
          // printf("tip = %d\n\n", pixels_shm->array.UI16[idx]);
          norm_flux += (float)masked;
        }
      }
      norm_flux = fabsf(norm_flux);

      // Pass 2: fill slopes directly
      uint32_t idx_slopes = 0;
      for (uint32_t i = 0; i < n_pix; i++) {
        for (uint32_t j = 0; j < n_pix; j++) {
          uint32_t idx = i * n_pix + j;
          if (mask_shm->array.UI16[idx] > 0) {
            float normalized = masked_image[idx] / norm_flux;
            float slope_val = normalized - ref_img_norm_shm->array.F[idx];

            if (idx_slopes < n_slopes) {
              slopes[idx_slopes++] = slope_val;
            }
          }
        }
      }

      // Matrix-vector multiply: computed_modes = RM_S2KL * slopes
      for (uint32_t i = 0; i < n_modes; i++) {
        modes_shm->array.F[i] = 0.0f;
        for (uint32_t j = 0; j < n_slopes; j++) {
          modes_shm->array.F[i] += S2M_shm->array.F[i * n_slopes + j] * slopes[j];
        }
      }
      // printf("tip = %f\n\n", modes_shm->array.F[0]);
      // printf("tip = %f\n\n", S2M_shm->array.F[0]);
      // printf("tip = %f\n\n", slopes[0]);
      // printf("tip = %f\n\n", mask_shm->array.F[120]);
      // printf("tip = %f\n\n",norm_flux);
      daoShmImagePart2ShmFinalize(modes_shm);
    }
  }

  // Cleanup
  free(slopes);
  free(masked_image);
  free(modes);

  free(pixels_shm);
  free(modes_shm);
  free(mask_shm);
  free(bias_image_shm);
  free(ref_img_norm_shm);
  free(S2M_shm);
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
  real_time_loop();


  return 0;
}
