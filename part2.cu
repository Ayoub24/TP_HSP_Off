//
// Created by ayoumabr93 on 12/12/22.
//
#include <stdio.h>
#include <stdlib.h>
#include <cassert>


// déclaration des matrices
float raw_data[32*32];
float C1_data[6*28*28];
float S1_data[6*14*14];
float C1_kernel[6*5*5];

// fonction d'initialisation des matrices
void init_raw_data(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

void init_C1_data(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 0;
    }
}

void init_S1_data(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 0;
    }
}

void init_C1_kernel(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = rand() / (float)RAND_MAX;
    }
}



void conv2d(float *input, float *filter, int input_width, int input_height, int filter_width, int filter_height, float *output)
{
    int output_width = input_width - filter_width + 1;
    int output_height = input_height - filter_height + 1;

    // boucle pour parcourir chaque pixel de l'image d'entrée
    for (int i = 0; i < output_height; i++)
    {
        for (int j = 0; j < output_width; j++)
        {
            // initialisation de la valeur de sortie à 0
            output[i * output_width + j] = 0;

            // boucle pour appliquer le filtre à chaque pixel de l'image
            for (int k = 0; k < filter_height; k++)
            {
                for (int l = 0; l < filter_width; l++)
                {
                    // calcul de la convolution en multipliant chaque élément du filtre par la valeur correspondante de l'image d'entrée et en les additionnant
                    output[i * output_width + j] += filter[k * filter_width + l] * input[(i + k) * input_width + (j + l)];
                }
            }
        }
    }
}

void sub_sampling_2D(float* input, int rows, int cols, float* output) {
    // Vérifier que les tailles de l'entrée et de la sortie sont correctes
    assert(rows % 2 == 0 && cols % 2 == 0);

    // Échantillonner l'entrée pour générer la sortie
    for (int i = 0; i < rows; i += 2) {
        for (int j = 0; j < cols; j += 2) {
            output[(i / 2)*(cols/2)+j / 2] = (input[i * cols + j] + input[i * cols + j + 1] + input[(i + 1) * cols + j] + input[(i + 1) * cols + j + 1]) / 4.0;
        }
    }
}

__device__ float activation_tanh(float M) {
    return tanhf(M);
}

__global__ void cudaMatrixMultGeneral(float *M1, float *M2, float *Mout, int n, int p, int m){

    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0f;

    if (lig < n && col < m){
        for (int i = 0; i < p; i++){
            s += M1[lig * p + i] * M2[i * m + col];
        }
        Mout[lig * m + col] = s;
    }
}


