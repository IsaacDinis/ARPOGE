#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <sched.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <libgen.h>
#include <errno.h>
#include "dao.h"
#include "fitsio.h"

#define VERBOSE 0

#define SHM_NAME_SIZE 32
#define N_MODES 195
#define N_ACT 241
#define ORDER 20
#define PRINT_RATE 1.0
#define N_ITER 5000
#define BOOTSTRAP_N_ITER 10
#define MAX_PATH 1024


char in_shm_name[SHM_NAME_SIZE] = "/tmp/papyrus_modes.im.shm";
// char in_shm_name[SHM_NAME_SIZE] = "/tmp/modes.shm";

char out_shm_name[SHM_NAME_SIZE] = "/tmp/dmCmd03.im.shm";
char M2V_shm_name[SHM_NAME_SIZE] = "/tmp/m2c.im.shm";

const int sem_nb = 7;

double get_time_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;  // casts are important
}


// Write double array to FITS file CHAT GPT
int write_fits_double(const char *filename, double *array, long n) { 
    fitsfile *fptr; 
    int status = 0; 
    long naxes[1] = {n}; 
    if (fits_create_file(&fptr, filename, &status)) 
        return status; 
    if (fits_create_img(fptr, DOUBLE_IMG, 1, naxes, &status)) 
        return status; 
    if (fits_write_img(fptr, TDOUBLE, 1, n, array, &status)) 
        return status; 
    if (fits_close_file(fptr, &status)) 
        return status; 
    return 0; 
}

// ---- mkdir -p equivalent ---- CHAT GPT
int mkdir_p(const char *path) {
    char tmp[MAX_PATH];
    strncpy(tmp, path, sizeof(tmp));
    tmp[sizeof(tmp) - 1] = '\0';

    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) && errno != EEXIST) return -1;
    return 0;
}

// ---- Save array to FITS ---- CHAT GPT
int save_array_to_fits(const char *array_name, double *array, long n_iter, long bootstrap_n_iter) {
    // 1) get executable directory
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1) {
        perror("readlink");
        return -1;
    }
    exe_path[len] = '\0';
    char *exe_dir = dirname(exe_path);   // e.g. dummy/build/apps

    // 2) go two levels up: dummy/build/apps → dummy/build → dummy
    char parent_dir[MAX_PATH];
    strncpy(parent_dir, exe_dir, sizeof(parent_dir) - 1);
    parent_dir[sizeof(parent_dir) - 1] = '\0';
    for (int i = 0; i < 2; i++) {
        char *slash = strrchr(parent_dir, '/');
        if (slash) *slash = '\0';
    }

    // 3) build results_dir = parent_dir + "/results"
    char results_dir[MAX_PATH];
    strncpy(results_dir, parent_dir, sizeof(results_dir) - 1);
    results_dir[sizeof(results_dir) - 1] = '\0';
    strncat(results_dir, "/results", sizeof(results_dir) - strlen(results_dir) - 1);

    // 4) timestamped folder
    char folder_name[64];
    time_t t = time(NULL);
    struct tm tm_info;
    localtime_r(&t, &tm_info);
    strftime(folder_name, sizeof(folder_name), "%Y-%m-%d_%H-%M-%S", &tm_info);
    strncat(folder_name, "_c", sizeof(folder_name) - strlen(folder_name) - 1);

    // 5) build full_path = results_dir + "/" + folder_name
    char full_path[MAX_PATH];
    strncpy(full_path, results_dir, sizeof(full_path) - 1);
    full_path[sizeof(full_path) - 1] = '\0';
    strncat(full_path, "/", sizeof(full_path) - strlen(full_path) - 1);
    strncat(full_path, folder_name, sizeof(full_path) - strlen(full_path) - 1);

    if (mkdir_p(full_path) != 0) {
        perror("mkdir_p");
        return -1;
    }

    // 6) slice array
    long n = n_iter - bootstrap_n_iter;
    double *array_ptr = &array[bootstrap_n_iter];

    // 7) build fits_file = full_path + "/" + array_name + ".fits"
    char fits_file[MAX_PATH];
    strncpy(fits_file, full_path, sizeof(fits_file) - 1);
    fits_file[sizeof(fits_file) - 1] = '\0';
    strncat(fits_file, "/", sizeof(fits_file) - strlen(fits_file) - 1);
    strncat(fits_file, array_name, sizeof(fits_file) - strlen(fits_file) - 1);
    strncat(fits_file, ".fits", sizeof(fits_file) - strlen(fits_file) - 1);

    if (write_fits_double(fits_file, array_ptr, n) != 0) {
        fprintf(stderr, "Error writing %s\n", fits_file);
        return -1;
    }

    printf("Saved %s to %s\n", array_name, full_path);
    return 0;
}

int main() {

  int RT_priority = 93; //any number from 0-99
  struct sched_param schedpar;

  schedpar.sched_priority = RT_priority;
  // r = seteuid(euid_called); //This goes up to maximum privileges
  sched_setscheduler(0, SCHED_FIFO, &schedpar); //other option is SCHED_RR, might be faster
  // r = seteuid(euid_real);//Go back to normal privileges

  const float gain = 0.3;

  IMAGE *in_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *out_shm = (IMAGE*) malloc(sizeof(IMAGE));
  IMAGE *M2V_shm = (IMAGE*) malloc(sizeof(IMAGE));

  float* state_mat = calloc((2 * ORDER + 1) * N_MODES, sizeof(float));
  float* K_mat     = calloc((2 * ORDER + 1) * N_MODES, sizeof(float));
  float* command   = calloc(N_MODES, sizeof(float));

  daoShmShm2Img(M2V_shm_name, &M2V_shm[0]);
  daoShmShm2Img(in_shm_name, &in_shm[0]);
  daoShmShm2Img(out_shm_name, &out_shm[0]);

  double* loop_time_array = calloc((int)(N_ITER), sizeof(double));
  double* wfs_timestamp_array = calloc((int)(N_ITER), sizeof(double));

  double last_loop_time = get_time_seconds();
  double computation_time = 0, wfs_time = 0;

  #if VERBOSE
    int counter = 0;
    double time_at_last_print = get_time_seconds();
  #endif

  for (int i = 0; i < N_MODES; i++) {
      K_mat[i] = gain; 
      K_mat[(ORDER + 1) * N_MODES + i] = 0.99f;
  }

  for (int iter = 0; iter < N_ITER; iter++) {
    
    double now = get_time_seconds();
    double dt = now - last_loop_time;
    loop_time_array[iter] = dt;
    last_loop_time = now;

    double start_wfs = get_time_seconds();
    daoShmWaitForSemaphore(in_shm, sem_nb);
    wfs_timestamp_array[iter] = (double)in_shm[0].md[0].atime.ts.tv_sec + (double)in_shm[0].md[0].atime.ts.tv_nsec / 1e9;

    wfs_time += get_time_seconds() - start_wfs;

    double start_compute = get_time_seconds();
    out_shm[0].md[0].cnt2 = in_shm[0].md[0].cnt2;

    memmove(&state_mat[N_MODES], state_mat, N_MODES * (2 * ORDER) * sizeof(float));
    memcpy(&state_mat[0], in_shm[0].array.F, N_MODES * sizeof(float));  // insert new row at top

    
    for (int i = 0; i < N_MODES; i++) {
        command[i] = 0;
        for (int j = 0; j < 2 * ORDER + 1; j++) {
            command[i] += state_mat[j * N_MODES + i] * K_mat[j * N_MODES + i];
        }
    }
    memcpy(&state_mat[ORDER * N_MODES], command, N_MODES * sizeof(float));

    for (int i = 0; i < N_ACT; i++) {
        out_shm[0].array.F[i] = 0;
        for (int j = 0; j < N_MODES; j++) {
            out_shm[0].array.F[i] -= M2V_shm[0].array.F[i * N_MODES + j] * command[j];
        }
    }
    daoShmImagePart2ShmFinalize(&out_shm[0]);
    computation_time += get_time_seconds() - start_compute;

    #if VERBOSE
        // ---- Periodic logging
        if (get_time_seconds() - time_at_last_print > PRINT_RATE) {
                // ---- Compute mean loop time ----
            double sum = 0.;
            for (int k = iter - counter; k < iter; k++) {
                sum += loop_time_array[k];
            }
            double loop_time_mean = sum / counter;

            // ---- Compute max ----
            double max_val = loop_time_array[iter - counter];
            for (int k = iter - counter; k < iter; k++) {
                if (loop_time_array[k] > max_val) {
                    max_val = loop_time_array[k];
                }
            }

            // ---- Count frame misses (loop_time > 2*mean) ----
            int frame_missed = 0;
            for (int k = iter - counter; k < iter; k++) {
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
            computation_time = counter = wfs_time = 0;
            time_at_last_print = get_time_seconds();
        }
            counter++;
    #endif
  }


    save_array_to_fits("loop_time_array", loop_time_array, N_ITER, BOOTSTRAP_N_ITER);
    save_array_to_fits("wfs_timestamp_array", wfs_timestamp_array, N_ITER, BOOTSTRAP_N_ITER);

  return 0;
}