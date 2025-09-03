#ifndef UTILS_H
#define UTILS_H
#include "toml.h"

#define MAX_PATH 1024
int write_fits_double_1d(const char *filename, double *array, long n);
int mkdir_p(const char *path) ;
int save_array_to_fits_1d(const char *rel_folder,
                       double *array, long n);
int save_array_to_fits_2d(const char *rel_folder,
                         double *array, long nx, long ny);
int write_fits_double_2d(const char *filename, double *array, long nx, long ny);
double* read_fits_1d(const char *filename, long *n_out);
double *read_fits_2d(const char *filename, long *nx_out, long *ny_out);
char* relative_path_write(const char *rel_folder);
char* relative_path_read(const char *rel_path);
toml_table_t* load_toml(const char *rel_path, char *errbuf, size_t errlen);
#endif /* UTILS_H */