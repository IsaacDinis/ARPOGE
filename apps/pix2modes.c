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

#define TIME_VERBOSE 1
#define PRINT_RATE 5
#define MAX_FS 1000

// ---------- Structures ----------

struct {
    char *modes;
    char *pixels;
    char *mask;
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
    if (sem_nb)    toml_rtoi(toml_raw_in(sem_nb, "pix2modes"),          &config.sem_nb);

    toml_free(root);
    return 0;
}

void free_shm_path() {
    free(shm_path.modes);
    free(shm_path.pixels);
    free(shm_path.mask);
    free(shm_path.ref_img_norm);
    free(shm_path.S2M);
}

int real_time_loop(){
  signal(SIGINT, endme);
  struct timespec timeout;

  IMAGE *pixels_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *modes_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *mask_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *ref_img_norm_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *S2M_shm = (IMAGE*) malloc(sizeof(IMAGE));

  daoShmShm2Img(shm_path.modes, modes_shm);
  daoShmShm2Img(shm_path.pixels, pixels_shm);
  daoShmShm2Img(shm_path.mask, mask_shm);
  daoShmShm2Img(shm_path.ref_img_norm, ref_img_norm_shm);
  daoShmShm2Img(shm_path.S2M, S2M_shm);

  uint32_t n_pix = pixels_shm->md[0].size[0] ;
  uint32_t n_slopes = S2M_shm->md[0].size[1];
  uint32_t n_modes = S2M_shm->md[0].size[0];

  float* modes  = calloc(n_modes, sizeof(float));
  float* slopes = malloc(n_slopes * sizeof(float));
  float* masked_image = malloc(n_pix * n_pix * sizeof(float));
  float norm_flux = 0.0f;

  #if TIME_VERBOSE
    double* loop_time_array = calloc((int)(PRINT_RATE*MAX_FS), sizeof(double));
    double last_loop_time = get_time_seconds();
    double computation_time = 0, wfs_time = 0;
    int counter = 0;
    double time_at_last_print = get_time_seconds();
    double start_wfs, start_compute, now, dt;
  #endif

  while(!end){
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += 1; // 1 second timeout

    #if TIME_VERBOSE
      now = get_time_seconds();
      dt = now - last_loop_time;
      loop_time_array[counter] = dt;
      last_loop_time = now;
      start_wfs = get_time_seconds();
    #endif

    if (daoShmWaitForSemaphoreTimeout(pixels_shm, config.sem_nb, &timeout) != -1){
      #if TIME_VERBOSE
        wfs_time += get_time_seconds() - start_wfs;
        start_compute = get_time_seconds();
      #endif

      modes_shm->md[0].cnt2 = pixels_shm->md[0].cnt2;

      norm_flux = 0.0f;
      for (uint32_t i = 0; i < n_pix; i++) {
        for (uint32_t j = 0; j < n_pix; j++) {
          uint32_t idx = i * n_pix + j;
          float corrected = (float)pixels_shm->array.UI16[idx];
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

      // start_compute = get_time_seconds();
      // Matrix-vector multiply: computed_modes = RM_S2KL * slopes
      for (uint32_t i = 0; i < n_modes; i++) {
        modes_shm->array.F[i] = 0.0f;
        for (uint32_t j = 0; j < n_slopes; j++) {
          modes_shm->array.F[i] += S2M_shm->array.F[i * n_slopes + j] * slopes[j];
        }
      }

      daoShmImagePart2ShmFinalize(modes_shm);

      #if TIME_VERBOSE
        computation_time += get_time_seconds() - start_compute;

        // ---- Periodic logging
        if (get_time_seconds() - time_at_last_print > PRINT_RATE) {
                // ---- Compute mean loop time ----
            double sum = 0.;
            for (int k = 0; k < counter; k++) {
                sum += loop_time_array[k];
            }
            double loop_time_mean = sum / counter;

            // ---- Compute max ----
            double max_val = loop_time_array[0];
            for (int k = 0; k < counter; k++) {
                if (loop_time_array[k] > max_val) {
                    max_val = loop_time_array[k];
                }
            }

            // ---- Count frame misses (loop_time > 2*mean) ----
            int frame_missed = 0;
            for (int k = 0; k < counter; k++) {
                if (loop_time_array[k] > 2.0 * loop_time_mean) {
                    frame_missed++;
                }
            }

            // ---- Print results ----
            printf("Mean Loop rate = %.2f Hz\n", 1.0 / loop_time_mean);
            printf("Mean Loop time = %.2f ms\n", loop_time_mean * 1e3);
            printf("Mean WFS time = %.2f ms\n", (wfs_time / counter) * 1e3);
            printf("Mean Computation time = %.2f ms\n", (computation_time / counter) * 1e3);
            printf("Max loop time = %.2f ms\n", max_val * 1e3);
            printf("Frames missed = %d\n\n", frame_missed);

            // Reset counters
            computation_time = wfs_time = 0;
            counter = -1;
            time_at_last_print = get_time_seconds();
        }
        counter++;
      #endif
    }
    else printf("Not receiving frames ! \n");
  }

  // Cleanup
  free(slopes);
  free(masked_image);
  free(modes);
  free(pixels_shm);
  free(modes_shm);
  free(mask_shm);
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
  free_shm_path();

  return 0;
}
