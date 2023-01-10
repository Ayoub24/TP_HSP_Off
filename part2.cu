#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS 32
#define COLS 32
#define C1_KERNEL_ROWS 5
#define C1_KERNEL_COLS 5
#define S1_ROWS 6
#define S1_COLS 14
#define C1_ROWS 28
#define C1_COLS 28

void convolution2D(float **image, float **filter, float **result, int image_rows, int image_cols, int filter_rows, int filter_cols) {
    int filter_mid_rows = (int)filter_rows / 2;
    int filter_mid_cols = (int)filter_cols / 2;

    for (int i = filter_mid_rows; i < image_rows - filter_mid_rows; ++i) {
        for (int j = filter_mid_cols; j < image_cols - filter_mid_cols; ++j) {
            int sum = 0;

            for (int m = 0; m < filter_rows; ++m) {
                for (int n = 0; n < filter_cols; ++n) {
                    sum += image[i - filter_mid_rows + m][j - filter_mid_cols + n] * filter[m][n];
                }
            }

            result[i][j] = sum;
        }
    }
}

int main() {
    // Génération de la matrice raw_data
    /*float* raw_data[ROWS][COLS];
    srand(time(NULL));
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            raw_data[i][j] = (float) rand() / (float) (RAND_MAX);
        }
    }

    // Génération de la matrice C1_kernel
    float* C1_kernel[C1_KERNEL_ROWS][C1_KERNEL_COLS];
    srand(time(NULL));
    for (int i = 0; i < C1_KERNEL_ROWS; i++) {
        for (int j = 0; j < C1_KERNEL_COLS; j++) {
            C1_kernel[i][j] = (float) rand() / (float) (RAND_MAX);
        }
    }

    // Génération des matrices C1_data et S1_data
    float* C1_data[C1_ROWS][C1_COLS];
    float* S1_data[S1_ROWS][S1_COLS];
    for (int i = 0; i < C1_ROWS; i++) {
        for (int j = 0; j < C1_COLS; j++) {
            C1_data[i][j] = 0;
        }
    }
    for (int i = 0; i < S1_ROWS; i++) {
        for (int j = 0; j < S1_COLS; j++) {
            S1_data[i][j] = 0;
        }
    }

    convolution2D(raw_data,C1_kernel,C1_data,ROWS,COLS,5,5);*/
}
