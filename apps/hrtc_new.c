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
    char *modes_in_4sided;
    char *modes_in_custom;
    char *modes_in_3sided;
    char *pyramid_select;
    char *dm;
    char *M2V;
    char *V2M;
    char *reset_flag;
    char *controller_select;
    char *closed_loop_state_flag;
    char *K_mat_int;
    char *K_mat_dd;
    char *K_mat_omgi;
    char *K_mat_flag;
    char *pyramid_flag;
    char *n_modes_controlled;

} shm_path;

struct {
    int64_t sem_nb;
    int64_t n_modes;
    int64_t n_act;
    int64_t max_order;
    double  max_voltage;
} config;

// ---------- Helpers ----------
static int end = 0;               // termination flag
// termination function for SIGINT callback
static void endme(){
    end = 1;
}

int load_shm_path() {
    char errbuf[200];
    toml_table_t *root = load_toml("../../config/shm_path.toml", errbuf, sizeof(errbuf));
    if (!root) return 1;

    toml_table_t *HW = toml_table_in(root, "HW");
    toml_table_t *settings = toml_table_in(root, "settings");
    toml_table_t *KL_mat = toml_table_in(root, "KL_mat");
    toml_table_t *K = toml_table_in(root, "K");
    toml_table_t *event_flag = toml_table_in(root, "event_flag");

    if (HW) {
        toml_rtos(toml_raw_in(HW, "modes_in_4sided"), &shm_path.modes_in_4sided);  
        toml_rtos(toml_raw_in(HW, "modes_in_custom"), &shm_path.modes_in_custom);
        toml_rtos(toml_raw_in(HW, "modes_in_3sided"), &shm_path.modes_in_3sided);
        toml_rtos(toml_raw_in(HW, "dm"),              &shm_path.dm);
    }
    
    if (KL_mat){
      toml_rtos(toml_raw_in(KL_mat, "M2V"),  &shm_path.M2V);
      toml_rtos(toml_raw_in(KL_mat, "V2M"),  &shm_path.V2M);
    } 
    
    if (settings) {
        toml_rtos(toml_raw_in(settings, "pyramid_select"),    &shm_path.pyramid_select);
        toml_rtos(toml_raw_in(settings, "controller_select"), &shm_path.controller_select);
        toml_rtos(toml_raw_in(settings, "closed_loop_state_flag"),  &shm_path.closed_loop_state_flag);
        toml_rtos(toml_raw_in(settings, "n_modes_controlled"),  &shm_path.n_modes_controlled);
    }

    if (K) {
        toml_rtos(toml_raw_in(K, "K_mat_int"),  &shm_path.K_mat_int);
        toml_rtos(toml_raw_in(K, "K_mat_dd"),   &shm_path.K_mat_dd);
        toml_rtos(toml_raw_in(K, "K_mat_omgi"), &shm_path.K_mat_omgi);
    }
    if (event_flag) {
        toml_rtos(toml_raw_in(event_flag, "reset_flag"),   &shm_path.reset_flag);
        toml_rtos(toml_raw_in(event_flag, "K_mat_flag"),   &shm_path.K_mat_flag);
        toml_rtos(toml_raw_in(event_flag, "pyramid_flag"), &shm_path.pyramid_flag);
    }

    toml_free(root);
    return 0;
}

int load_config() {
    char errbuf[200];
    toml_table_t *root = load_toml("../../config/config.toml", errbuf, sizeof(errbuf));
    if (!root) return 1;

    toml_table_t *common    = toml_table_in(root, "common");
    toml_table_t *hrtc      = toml_table_in(root, "hrtc");
    toml_table_t *sem_nb    = toml_table_in(root, "sem_nb");
    toml_table_t *optimizer = toml_table_in(root, "optimizer"); 

    if (common){    
      toml_rtoi(toml_raw_in(common, "n_modes"), &config.n_modes);
      toml_rtoi(toml_raw_in(common, "n_act"),   &config.n_act);
    }

    if (hrtc)      toml_rtod(toml_raw_in(hrtc, "max_voltage"),     &config.max_voltage);
    if (sem_nb)    toml_rtoi(toml_raw_in(sem_nb, "hrtc"),          &config.sem_nb);
    if (optimizer) toml_rtoi(toml_raw_in(optimizer, "max_order"),  &config.max_order);

    toml_free(root);
    return 0;
}

// ---------- Free helpers ----------

void free_shm_path() {
    free(shm_path.modes_in_4sided);
    free(shm_path.modes_in_custom);
    free(shm_path.modes_in_3sided);
    free(shm_path.dm);
    free(shm_path.pyramid_select);
    free(shm_path.reset_flag);
    free(shm_path.controller_select);
    free(shm_path.closed_loop_state_flag);
    free(shm_path.M2V);
    free(shm_path.V2M);
    free(shm_path.K_mat_int);
    free(shm_path.K_mat_dd);
    free(shm_path.K_mat_omgi);
    free(shm_path.K_mat_flag);
    free(shm_path.pyramid_flag);
}

// int load_K_mat(float* K_mat){
int load_K_mat(IMAGE *K_mat_shm, uint32_t controller_select){
  
  enum{INTEGRATOR,OMGI,DD};
  switch (controller_select) {
    case INTEGRATOR:
      daoShmShm2Img(shm_path.K_mat_int, K_mat_shm);
      break;
    case OMGI:
      daoShmShm2Img(shm_path.K_mat_omgi, K_mat_shm);
      break;
    case DD:
      daoShmShm2Img(shm_path.K_mat_dd, K_mat_shm);
      break;
    default:
      printf("This controller type does not exist");
      return 1;
  }
  return 0;
}

int select_pyramid(IMAGE *modes_in_shm, uint32_t pyramid_select){
  enum{PYR_4_SIDED,PYR_3_SIDED,CUSTOM};

  switch (pyramid_select) {
    case PYR_4_SIDED:
      daoShmShm2Img(shm_path.modes_in_4sided, modes_in_shm);
      break;
    case PYR_3_SIDED:
      daoShmShm2Img(shm_path.modes_in_3sided, modes_in_shm);
      break;
    case CUSTOM:
      daoShmShm2Img(shm_path.modes_in_custom, modes_in_shm);
      break;
    default:
      printf("This pyramid type does not exist");
      return 1;
  }
  return 0;
}

int check_flag(IMAGE *flag_shm){
  if(flag_shm->array.UI32[0]){
    flag_shm->array.UI32[0] = 0;
    return 1;
  } 
  return 0;
}

void reset_state_mat(float* state_mat){
  size_t len = (2 * config.max_order + 1) * config.n_modes;
  memset(state_mat, 0, len * sizeof(float));
}

static inline float clamp(float x, float max, bool *saturated) {
  if (x < -max) {
    *saturated = true;
    return -max;
  } 
  else if (x > max) {
    *saturated = true;
    return max;
  }
  return x;
}

int real_time_loop(){

  signal(SIGINT, endme);

  float* state_mat = calloc((2 * config.max_order + 1) * config.n_modes, sizeof(float));
  float* modes_out = calloc(config.n_modes, sizeof(float));

  bool saturated = 0;

  IMAGE *modes_in_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *dm_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *M2V_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *V2M_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *K_mat_flag_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *reset_flag_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *pyramid_flag_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *closed_loop_state_flag_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *K_mat_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *controller_select_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *pyramid_select_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *n_modes_controlled_shm = (IMAGE*) malloc(sizeof(IMAGE));
  daoShmShm2Img(shm_path.pyramid_select, pyramid_select_shm);
  daoShmShm2Img(shm_path.controller_select, controller_select_shm);
  daoShmShm2Img(shm_path.dm, dm_shm);
  daoShmShm2Img(shm_path.M2V, M2V_shm);
  daoShmShm2Img(shm_path.V2M, V2M_shm);
  daoShmShm2Img(shm_path.K_mat_flag, K_mat_flag_shm);
  daoShmShm2Img(shm_path.reset_flag, reset_flag_shm);
  daoShmShm2Img(shm_path.pyramid_flag, pyramid_flag_shm);
  daoShmShm2Img(shm_path.closed_loop_state_flag, closed_loop_state_flag_shm);
  daoShmShm2Img(shm_path.n_modes_controlled, n_modes_controlled_shm);

  load_K_mat(K_mat_shm,controller_select_shm->array.UI32[0]);
  select_pyramid(modes_in_shm, pyramid_select_shm->array.UI32[0]);

  while(!end){

    if(check_flag(K_mat_flag_shm))load_K_mat(K_mat_shm,controller_select_shm->array.UI32[0]);
    if(check_flag(pyramid_flag_shm))select_pyramid(modes_in_shm, pyramid_select_shm->array.UI32[0]);
    if(check_flag(reset_flag_shm)) reset_state_mat(state_mat);
    saturated = 0;
    daoShmWaitForSemaphore(modes_in_shm, config.sem_nb);

    dm_shm->md[0].cnt2 = modes_in_shm->md[0].cnt2;

    memmove(&state_mat[config.n_modes], state_mat, config.n_modes * (2 * config.max_order) * sizeof(float));
    memcpy(&state_mat[0], modes_in_shm->array.F, config.n_modes * sizeof(float)); 
    
    for (uint32_t i = 0; i < (uint32_t) config.n_modes; i++) {
      modes_out[i] = 0;
      if(closed_loop_state_flag_shm->array.UI32[0]&&i<n_modes_controlled_shm->array.UI32[0]){
        for (int j = 0; j < 2 * config.max_order + 1; j++) {
          modes_out[i] += state_mat[j * config.n_modes + i] * K_mat_shm->array.F[j * config.n_modes + i];
        }
      }
    }
    for (int i = 0; i < config.n_act; i++) {
        dm_shm->array.F[i] = 0;
        for (int j = 0; j < config.n_modes; j++) {
            dm_shm->array.F[i] -= M2V_shm->array.F[i * config.n_modes + j] * modes_out[j];
        }
        dm_shm->array.F[i] = clamp(dm_shm->array.F[i], config.max_voltage, &saturated);
    }
    if (saturated) {
      for (int j = 0; j < config.n_modes; j++) {
          modes_out[j] = 0.0f;
          for (int i = 0; i < config.n_act; i++) {
              modes_out[j] -= V2M_shm->array.F[j * config.n_act + i] * dm_shm->array.F[i];
          }
      }
    }
    memcpy(&state_mat[config.max_order * config.n_modes], modes_out, config.n_modes * sizeof(float));

    daoShmImagePart2ShmFinalize(dm_shm);
  }

  free(modes_out);
  free(state_mat);
  free(modes_in_shm);
  free(dm_shm);
  free(M2V_shm);
  free(V2M_shm);
  free(K_mat_flag_shm);
  free(reset_flag_shm);
  free(pyramid_flag_shm);
  free(closed_loop_state_flag_shm);
  free(K_mat_shm);
  free(controller_select_shm);
  free(pyramid_select_shm);
  free(n_modes_controlled_shm);
  return 0;
}

// ---------- Main ----------

int main(void) {
  int RT_priority = 93; //any number from 0-99
  struct sched_param schedpar;

  schedpar.sched_priority = RT_priority;
  // r = seteuid(euid_called); //This goes up to maximum privileges
  sched_setscheduler(0, SCHED_FIFO, &schedpar); //other option is SCHED_RR, might be faster
  // r = seteuid(euid_real);//Go back to normal privileges
  load_config();
  load_shm_path();
  real_time_loop();
  free_shm_path();
  return 0;
}
