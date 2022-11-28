#include <iostream>

void MatrixInit(float *M, int n, int p){
    for(int row=0; row<n; row++){
        for (int col=0; col<p;  col++){
            M[row*p+col]=float(rand())/(float((RAND_MAX)/2))-1;
        }
    }
}


void MatrixPrint(float *M, int n, int p){
    printf("[");
    for (int row=0; row<n; row++)
    {
        for(int col=0; col<p; col++)
        {
            if (row==n-1 & col==p-1){
                printf("%f]", M[row*p+col]);
            }else{
                printf("%f    ", M[row*p+col]);
            }
        }
        printf("\n");
    }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < p; col++) {
            Mout[row*p+col] = M1[row*p+col] + M2[row*p+col];
        }
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){

}


void MatrixMult(float *M1, float *M2, float *Mout, int n){
    float sum=0;
    for(int i = 0; i<n; i++){
        for(int j=0; j<n; j++){
            sum=0;
            for(int k=0; k<n; k++){
                sum += M1[i*n+k]*M2[k*n+j];
            }
            Mout[i*n+j] = sum;
        }
    }
}

void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){

}

int main() {
    std::cout << "Hello, World!" << std::endl;

    int n = 5;
    int p = 5;
    float *M1;
    float *M2;
    float *M3;
    float *Mtest1;
    float *Mtest2;
    M1 = (float *) malloc(sizeof(float)*p*n);
    M2 = (float *) malloc(sizeof(float)*p*n);
    M3 = (float *) malloc(sizeof(float)*p*n);
    Mtest1 = (float *) malloc(sizeof(float)*n*n);
    Mtest2 = (float *) malloc(sizeof(float)*n*n);
    MatrixInit(M1,n,p);
    MatrixInit(M2,n,p);
    MatrixInit(Mtest1,n,n);
    MatrixInit(Mtest2,n,n);


    MatrixPrint(M1,n,p);
    printf("\n");
    MatrixPrint(M2,n,p);
    printf("\n");
    MatrixAdd(M1, M2, M3, n, p);
    MatrixPrint(M3,n,p);
    printf("\n");
    MatrixMult(Mtest1,Mtest2,M3,n);
    MatrixPrint(M3,n,n);
    printf("\n");


    return 0;
}
