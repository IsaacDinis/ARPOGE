#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "dao.h"
#include "toml.h"
#include "utils.h"

// ---------- Structures ----------
#define SHM_NAME_SIZE 32
#define N_MODES 195
#define N_ACT 241
#define ORDER 20
#define PRINT_RATE 1.0
#define N_ITER 5000
#define BOOTSTRAP_N_ITER 10
#define MAX_PATH 1024

typedef struct {
    char *modes_in_4sided;
    char *modes_in_custom;
    char *modes_in_3sided;
    char *pyramid_select;
    char *dm;
    char *M2V;
    char *reset_flag;
    char *controller_select;
    char *closed_loop_flag;
    char *K_mat_int;
    char *K_mat_dd;
    char *K_mat_omgi;

} shm_path_t;

struct {
    int64_t sem_nb;
    int64_t n_modes;
    int64_t n_act;
    int64_t max_order;
    double  max_voltage;
} config;

// ---------- Helpers ----------


int load_shm_path(shm_path_t *shm_path_p) {
    char errbuf[200];
    toml_table_t *root = load_toml("../../config/shm_path.toml", errbuf, sizeof(errbuf));
    if (!root) return 1;

    toml_table_t *HW = toml_table_in(root, "HW");
    toml_table_t *settings = toml_table_in(root, "settings");
    toml_table_t *KL_mat = toml_table_in(root, "KL_mat");
    toml_table_t *K = toml_table_in(root, "K");

    if (HW) {
        toml_rtos(toml_raw_in(HW, "modes_in_4sided"), &shm_path_p->modes_in_4sided);  
        toml_rtos(toml_raw_in(HW, "modes_in_custom"), &shm_path_p->modes_in_custom);
        toml_rtos(toml_raw_in(HW, "modes_in_3sided"), &shm_path_p->modes_in_3sided);
        toml_rtos(toml_raw_in(HW, "dm"),              &shm_path_p->dm);
    }
    
    if (KL_mat) toml_rtos(toml_raw_in(KL_mat, "M2V"),  &shm_path_p->M2V);
    
    if (settings) {
        toml_rtos(toml_raw_in(settings, "pyramid_select"),    &shm_path_p->pyramid_select);
        toml_rtos(toml_raw_in(settings, "reset_flag"),        &shm_path_p->reset_flag);
        toml_rtos(toml_raw_in(settings, "controller_select"), &shm_path_p->controller_select);
        toml_rtos(toml_raw_in(settings, "closed_loop_flag"),  &shm_path_p->closed_loop_flag);
    }

    if (K) {
        toml_rtos(toml_raw_in(K, "K_mat_int"),  &shm_path_p->K_mat_int);
        toml_rtos(toml_raw_in(K, "K_mat_dd"),   &shm_path_p->K_mat_dd);
        toml_rtos(toml_raw_in(K, "K_mat_omgi"), &shm_path_p->K_mat_omgi);
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

void free_shm_path(shm_path_t *shm_path_p) {
    free(shm_path_p->modes_in_4sided);
    free(shm_path_p->modes_in_custom);
    free(shm_path_p->modes_in_3sided);
    free(shm_path_p->dm);
    free(shm_path_p->pyramid_select);
    free(shm_path_p->reset_flag);
    free(shm_path_p->controller_select);
    free(shm_path_p->closed_loop_flag);
    free(shm_path_p->M2V);
    free(shm_path_p->K_mat_int);
    free(shm_path_p->K_mat_dd);
    free(shm_path_p->K_mat_omgi);
}

int compute_voltages(float* K_mat, shm_path_t shm_path){
  float* state_mat = calloc((2 * config.max_order + 1) * config.n_modes, sizeof(float));
  float* command   = calloc(config.n_modes, sizeof(float));
  IMAGE modes_in_shm;
  IMAGE dm_shm;
  IMAGE pyramid_select_shm;
  IMAGE M2V_shm;
  daoShmShm2Img(shm_path.pyramid_select, &pyramid_select_shm);
  daoShmShm2Img(shm_path.dm, &dm_shm);
  daoShmShm2Img(shm_path.M2V, &M2V_shm);
  enum{PYR_4_SIDED,PYR_3_SIDED,CUSTOM};
  switch (pyramid_select_shm.array.UI32[0]) {
    case PYR_4_SIDED:
      daoShmShm2Img(shm_path.modes_in_4sided, &modes_in_shm);
      break;
    case PYR_3_SIDED:
      daoShmShm2Img(shm_path.modes_in_3sided, &modes_in_shm);
      break;
    case CUSTOM:
      daoShmShm2Img(shm_path.modes_in_custom, &modes_in_shm);
      break;
    default:
      printf("This pyramid type does not exist");
      return 1;
  }
  while(1){
    daoShmWaitForSemaphore(&modes_in_shm, config.sem_nb);

    dm_shm.md[0].cnt2 = modes_in_shm.md[0].cnt2;

    memmove(&state_mat[N_MODES], state_mat, N_MODES * (2 * ORDER) * sizeof(float));
    memcpy(&state_mat[0], modes_in_shm.array.F, N_MODES * sizeof(float));  // insert new row at top
    
    for (int i = 0; i < N_MODES; i++) {
        command[i] = 0;
        for (int j = 0; j < 2 * ORDER + 1; j++) {
            command[i] += state_mat[j * N_MODES + i] * K_mat[j * N_MODES + i];
        }
    }
    memcpy(&state_mat[ORDER * N_MODES], command, N_MODES * sizeof(float));

    for (int i = 0; i < N_ACT; i++) {
        dm_shm.array.F[i] = 0;
        for (int j = 0; j < N_MODES; j++) {
            dm_shm.array.F[i] -= M2V_shm.array.F[i * N_MODES + j] * command[j];
        }
    }
    daoShmImagePart2ShmFinalize(&dm_shm);
  }
  return 0;
}

uint32_t check_pyramid_select(char *pyramid_select){
  // IMAGE *pyramid_select_shm = (IMAGE*) malloc(sizeof(IMAGE));
  // daoShmShm2Img(pyramid_select, &pyramid_select_shm[0]);
  // printf("pyramid select = %d\n",  pyramid_select_shm[0].array.UI32[0]);

  IMAGE pyramid_select_shm;
  daoShmShm2Img(pyramid_select, &pyramid_select_shm);
  // printf("pyramid select = %d\n", pyramid_select_shm.array.UI32[0]);
  return pyramid_select_shm.array.UI32[0];
}

// int load_K_mat(double *K_mat, char *controller_select){
int load_K_mat(float* K_mat, shm_path_t shm_path){
  
  IMAGE controller_select_shm;
  IMAGE K_mat_shm;
  daoShmShm2Img(shm_path.controller_select, &controller_select_shm);

  enum{INTEGRATOR,OMGI,DD};
  switch (controller_select_shm.array.UI32[0]) {
    case INTEGRATOR:
      daoShmShm2Img(shm_path.K_mat_int, &K_mat_shm);
      memcpy(K_mat, K_mat_shm.array.F, (2*config.max_order+1)*config.n_modes*sizeof(float));
      break;
    case OMGI:
      daoShmShm2Img(shm_path.K_mat_omgi, &K_mat_shm);
      memcpy(K_mat, K_mat_shm.array.F, (2*config.max_order+1)*config.n_modes*sizeof(float));
      break;
    case DD:
      daoShmShm2Img(shm_path.K_mat_dd, &K_mat_shm);
      memcpy(K_mat, K_mat_shm.array.F, (2*config.max_order+1)*config.n_modes*sizeof(float));
      break;
    default:
      printf("This controller type does not exist");
      return 1;
  }
  return 0;
}

// ---------- Main ----------

int main(void) {
  if (load_config() == 0) {
      printf("n_modes: %ld\n", config.n_modes);
      printf("max_voltage: %f\n", config.max_voltage);
      printf("sem_nb: %ld\n", config.sem_nb);
      printf("max_order: %ld\n", config.max_order);
  }
  float* K_mat     = calloc((2 * config.max_order + 1) * config.n_modes, sizeof(float));
  shm_path_t shm_path = {0};
  if (load_shm_path(&shm_path) == 0) {
      printf("modes_in_4sided: %s\n", shm_path.modes_in_4sided);
      printf("dm: %s\n", shm_path.dm);
      printf("M2V: %s\n", shm_path.M2V);
  }

  // config_t config = {0};

  // check_pyramid_select(shm_path.pyramid_select);
  printf("pyramid select = %d\n", check_pyramid_select(shm_path.pyramid_select));

  load_K_mat(K_mat,shm_path);
  compute_voltages(K_mat,shm_path);
  printf("K mat 0 = %f\n", K_mat[config.max_order]);
  // free_shm_path(&shm_path);
  // free(K_mat);
  return 0;
}
