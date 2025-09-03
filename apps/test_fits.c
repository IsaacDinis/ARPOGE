#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <sched.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <libgen.h>
#include <errno.h>
#include "utils.h"


int main() {
    
    double arr[] = {2, 4, 8, 12, 16, 18};
    double matrix[2][3] = { {1, 4, 2}, {3, 6, 8} };
    int n = sizeof(arr)/sizeof(arr[0]);
    n = sizeof(matrix)/sizeof(matrix[0][0]);
    // Printing array elements
    for (int i = 0; i < n; i++) {
        printf("%f ", arr[i]);
    }
    printf("%d ", n);
    save_array_to_fits_1d("../../results/test.fits",arr, n);
    save_array_to_fits_2d("../../results/test_2d.fits",&matrix[0][0],2,3);
    long nx, ny;
	double *matrix2 = read_fits_2d("../../results/test_2d.fits", &nx, &ny);

	for (long y=0; y<ny; y++) {
	    for (long x=0; x<nx; x++) {
	        printf("%f ", matrix2[y*nx + x]);
	    }
	    printf("\n");
	}
	free(matrix2);

	long n1;
	double *arr2 = read_fits_1d("../../results/test.fits", &n1);
	for (long i=0; i<n1; i++) printf("%f ", arr2[i]);
	free(arr2);

    return 0;
}