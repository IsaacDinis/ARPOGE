#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <sched.h>
#include "dao.h"
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <libgen.h>
#include "fitsio.h"
#include <errno.h>
#include <unistd.h>  
#include "utils.h"
#include <stdbool.h> 
#include "toml.h"


// Write double array to FITS file 
int write_fits_double_1d(const char *filename, double *array, long n) { 
    fitsfile *fptr; 
    int status = 0; 
    long naxes[1] = {n}; 

    remove(filename);

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


int write_fits_double_2d(const char *filename, double *array, long nx, long ny) {
    fitsfile *fptr;
    int status = 0;
    long naxes[2] = {ny, nx}; // rows, columns

    remove(filename);

    if (fits_create_file(&fptr, filename, &status)) return status;
    if (fits_create_img(fptr, DOUBLE_IMG, 2, naxes, &status)) return status;

    long nelements = nx * ny;
    if (fits_write_img(fptr, TDOUBLE, 1, nelements, array, &status)) return status;

    if (fits_close_file(fptr, &status)) return status;

    return 0;
}

// ---- mkdir -p equivalent ----
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

char* relative_path_write(const char *rel_path) {
    // 1) get executable directory
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1) {
        perror("readlink");
        return NULL;
    }
    exe_path[len] = '\0';
    char *exe_dir = dirname(exe_path);

    // 2) build full path = exe_dir + "/" + rel_path
    char file[PATH_MAX];
    strncpy(file, exe_dir, sizeof(file) - 1);
    file[sizeof(file) - 1] = '\0';
    strncat(file, "/", sizeof(file) - strlen(file) - 1);
    strncat(file, rel_path, sizeof(file) - strlen(file) - 1);

    // 3) extract directory part to check write permission
    char dir_path[PATH_MAX];
    strncpy(dir_path, file, sizeof(dir_path) - 1);
    dir_path[sizeof(dir_path) - 1] = '\0';
    char *dir = dirname(dir_path);

    struct stat st;
    if (stat(dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "Directory does not exist: %s\n", dir);
        return NULL;
    }

    if (access(dir, W_OK) != 0) {
        fprintf(stderr, "No write permission for directory: %s\n", dir);
        return NULL;
    }

    // 4) return a strdup'd version
    return strdup(file);
}

char* relative_path_read(const char *rel_path) {
    // 1) get executable directory
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1) {
        perror("readlink");
        return NULL;
    }
    exe_path[len] = '\0';
    char *exe_dir = dirname(exe_path);

    // 2) build full path = exe_dir + "/" + rel_path
    char file[PATH_MAX];
    strncpy(file, exe_dir, sizeof(file) - 1);
    file[sizeof(file) - 1] = '\0';
    strncat(file, "/", sizeof(file) - strlen(file) - 1);
    strncat(file, rel_path, sizeof(file) - strlen(file) - 1);

    // 3) check that file exists and is readable
    struct stat st;
    if (stat(file, &st) != 0 || !S_ISREG(st.st_mode)) {
        fprintf(stderr, "File does not exist: %s\n", file);
        return NULL;
    }
    if (access(file, R_OK) != 0) {
        fprintf(stderr, "No read permission for file: %s\n", file);
        return NULL;
    }

    // 4) return a strdup'd version
    return strdup(file);
}

int save_array_to_fits_1d(const char *rel_folder, double *array, long n) {
    char *fits_file = relative_path_write(rel_folder);
    if (!fits_file) {
        return -1;
    }

    if (write_fits_double_1d(fits_file, array, n) != 0) {
        fprintf(stderr, "Error writing %s\n", fits_file);
        free(fits_file);
        return -1;
    }

    printf("Saved %s \n", rel_folder);
    free(fits_file);
    return 0;
}

int save_array_to_fits_2d(const char *rel_folder, double *array, long nx, long ny) {
    char *fits_file = relative_path_write(rel_folder);
    if (!fits_file) {
        return -1;
    }
    if (write_fits_double_2d(fits_file, array, nx, ny) != 0) {
        fprintf(stderr, "Error writing %s\n", fits_file);
        free(fits_file);
    return -1;
    }

    printf("Saved %s (%ldx%ld) \n", rel_folder, nx, ny);
    free(fits_file);
    return 0;
}


double* read_fits_1d(const char *filename, long *n_out) {
    char *path = relative_path_read(filename);
    if (!path) {
        return NULL;
    }

    fitsfile *fptr;
    int status = 0, naxis;
    long naxes[1] = {1};
    double *array = NULL;

    if (fits_open_file(&fptr, path, READONLY, &status)) {
        fits_report_error(stderr, status);
        free(path);
        return NULL;
    }
    free(path);
    if (fits_get_img_dim(fptr, &naxis, &status) || naxis != 1) {
        fprintf(stderr, "Error: not a 1D FITS file\n");
        fits_close_file(fptr, &status);
        return NULL;
    }

    if (fits_get_img_size(fptr, 1, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return NULL;
    }

    array = malloc(sizeof(double) * naxes[0]);
    if (!array) {
        fprintf(stderr, "Memory allocation error\n");
        fits_close_file(fptr, &status);
        return NULL;
    }

    long fpixel = 1;
    if (fits_read_img(fptr, TDOUBLE, fpixel, naxes[0], NULL, array, NULL, &status)) {
        fits_report_error(stderr, status);
        free(array);
        fits_close_file(fptr, &status);
        return NULL;
    }

    fits_close_file(fptr, &status);
    *n_out = naxes[0];
    return array;
}

double *read_fits_2d(const char *filename, long *nx_out, long *ny_out) {
    char *path = relative_path_read(filename);
    if (!path) {
        return NULL;
    }

    fitsfile *fptr;
    int status = 0, naxis;
    long naxes[2] = {1, 1};
    double *array = NULL;

    if (fits_open_file(&fptr, path, READONLY, &status)) {
        fits_report_error(stderr, status);
        free(path);
        return NULL;
    }
    free(path);
    if (fits_get_img_dim(fptr, &naxis, &status) || naxis != 2) {
        fprintf(stderr, "Error: not a 2D FITS file\n");
        fits_close_file(fptr, &status);
        return NULL;
    }

    if (fits_get_img_size(fptr, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return NULL;
    }

    long nx = naxes[0]; // columns
    long ny = naxes[1]; // rows

    array = malloc(sizeof(double) * nx * ny);
    if (!array) {
        fprintf(stderr, "Memory allocation error\n");
        fits_close_file(fptr, &status);
        return NULL;
    }

    long fpixel = 1;
    if (fits_read_img(fptr, TDOUBLE, fpixel, nx*ny, NULL, array, NULL, &status)) {
        fits_report_error(stderr, status);
        free(array);
        fits_close_file(fptr, &status);
        return NULL;
    }

    fits_close_file(fptr, &status);
    *nx_out = nx;
    *ny_out = ny;
    return array; // row-major
}

toml_table_t* load_toml(const char *rel_path, char *errbuf, size_t errlen) {
    char *path = relative_path_read(rel_path);
    if (!path) return NULL;

    FILE *fp = fopen(path, "r");
    free(path);  // `relative_path_read` probably mallocs
    if (!fp) {
        perror("fopen");
        return NULL;
    }

    toml_table_t *tbl = toml_parse_file(fp, errbuf, errlen);
    fclose(fp);

    if (!tbl) fprintf(stderr, "TOML error: %s\n", errbuf);
    return tbl;
}

