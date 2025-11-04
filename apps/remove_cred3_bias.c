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
#define PRINT_RATE 1
#define MAX_FS 1000

// ---------- Structures ----------

struct {
    char *pixels;
    char *pixels_wo_bias;
    char *bias_image;
} shm_path;

struct {
    int64_t sem_nb;

} config;

static int end = 0;               // termination flag
// termination function for SIGINT callback
static void endme(){
    end = 1;
}


int load_shm_path() {
    char errbuf[200];
    toml_table_t *root = load_toml("../config/shm_path.toml", errbuf, sizeof(errbuf));
    if (!root) return 1;

    toml_table_t *HW = toml_table_in(root, "HW");
    if (HW) {
        toml_rtos(toml_raw_in(HW, "pixels_3sided"),   &shm_path.pixels);
        toml_rtos(toml_raw_in(HW, "pixels_wo_bias_3sided"),   &shm_path.pixels_wo_bias);
    }
    toml_table_t *calibration = toml_table_in(root, "calibration");
    if (calibration) {
        toml_rtos(toml_raw_in(calibration, "bias_image"),   &shm_path.bias_image);
    }
    toml_free(root);
    return 0;
}

int load_config() {
    char errbuf[200];
    toml_table_t *root = load_toml("../config/config.toml", errbuf, sizeof(errbuf));
    if (!root) return 1;

    toml_table_t *sem_nb    = toml_table_in(root, "sem_nb");
    if (sem_nb)    toml_rtoi(toml_raw_in(sem_nb, "pix2modes"),          &config.sem_nb);

    toml_free(root);
    return 0;
}

void free_shm_path() {
    free(shm_path.pixels);
    free(shm_path.pixels_wo_bias);
    free(shm_path.bias_image);
}

int real_time_loop(){
  signal(SIGINT, endme);
  struct timespec timeout;
  IMAGE *pixels_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *pixels_wo_bias_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *bias_image_shm = (IMAGE*) malloc(sizeof(IMAGE));
  daoShmShm2Img(shm_path.pixels, pixels_shm);
  daoShmShm2Img(shm_path.pixels_wo_bias, pixels_wo_bias_shm);
  daoShmShm2Img(shm_path.bias_image, bias_image_shm);
  uint32_t n_pix_x = pixels_shm->md[0].size[1] ;
  uint32_t n_pix_y = pixels_shm->md[0].size[0] ;
  uint64_t cred3_frame_cnt = 0;


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
      cred3_frame_cnt = (uint64_t) pixels_shm->array.UI16[0]; // first pixel is the counter
      pixels_wo_bias_shm->md[0].cnt2 = cred3_frame_cnt;

      for (uint32_t i = 0; i < n_pix_x; i++) {
        for (uint32_t j = 0; j < n_pix_y; j++) {
          uint32_t idx = i * n_pix_y + j;
          if (idx) pixels_wo_bias_shm->array.F[idx] = (float)pixels_shm->array.UI16[idx] - bias_image_shm->array.F[idx];
          else pixels_wo_bias_shm->array.F[idx] = 0; // set first pixel to 0 due to pixel counter
        }
      }

      daoShmImagePart2ShmFinalize(pixels_wo_bias_shm);
      

      #if TIME_VERBOSE
        computation_time += get_time_seconds() - start_compute;
      #endif


      #if TIME_VERBOSE
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
  free(pixels_wo_bias_shm);
  free(pixels_shm);
  free(bias_image_shm);

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