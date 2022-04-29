#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <fstream>

#define myInt64 signed __int64
#define INT32 unsigned __int32

#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)
using namespace std;
using namespace chrono;

typedef union
{       myInt64 int64;
        struct {INT32 lo, hi;} int32;
} tsc_counter;

void kernel( int sizeKernel, int sizeMatrix, int sizeResult, double* matrix, double* result, double* filter ) {
    /*
     * Assumptions:
     * matrix - stored row-wise
     * res - stored row-wise
     * filter - stored row-wise
    */

    for (int i = 0; i < sizeResult; i++) {
        for (int j = 0; j < sizeResult; j++) {
            for (int kx = 0; kx < sizeKernel; kx++) {
                for (int ky = 0; ky < sizeKernel; ky++) {
                    result[i * sizeResult + j] += matrix[(i * sizeMatrix)+ j + kx + ky] * filter[kx * sizeKernel + ky];
                }
            }
        }
    }
}


int main(int argc, char **argv){
    ofstream out("logs_conv.txt", ios::app);
    int sizeMatrix = 1024;
    for(int sizeMatrix=1024;sizeMatrix<=16384;sizeMatrix*=2)
    {
        int sizeKernel = 4;
        int padding = 0;
        int strides = 1;
        int runs = 10;


        double *matrix ;
        double *filter ;
        double *result ;
        double sum = 0;
        tsc_counter t0, t1;

        int sizeResult = (((sizeMatrix - sizeKernel + 2 * padding) / strides) + 1);
        printf("size Result: %d\n", sizeResult);

        posix_memalign((void**) &matrix, 64, sizeMatrix * sizeMatrix * sizeof(double));
        posix_memalign((void**) &filter, 64, sizeKernel * sizeKernel * sizeof(double));
        posix_memalign((void**) &result, 64, sizeResult * sizeResult * sizeof(double));

        for (int i = 0; i != 16; ++i) {
            filter[i] = 2;
        }

        for(int i = 0; i<sizeMatrix*sizeMatrix;++i) {
                matrix[i] = 1.0;
        }

        for(int i = 0; i<sizeResult*sizeResult;++i) {
            result[i] = 0.0;
        }

        for(unsigned int i = 0; i != runs; ++i) {
            auto time_start = system_clock::now();
            kernel(sizeKernel,sizeMatrix,sizeResult,matrix,result,filter);
            auto time_end = system_clock::now();
            auto duration = duration_cast<microseconds>(time_end - time_start);
            sum += duration.count();
        }

        printf("Average time: %lf\n", ((double) (sum / 1000.0 / ((double) runs))));
        out << "conv" << '\t' << sizeMatrix  << '\t' << (double) (sum / 1000.0 / ((double) runs)) << endl;
        //free(matrix);
        //free(result);
        //free(filter);
    }
    out.close();
	system("pause");

/**
 * To check correctness uncomment the below code
 */

    // for(int i = 0; i<sizeResult;i++){
    //     for(int j= 0; j<sizeResult;j++){
    //        printf("%f           ",result[i*sizeResult + j]);
    //     }
    //     printf("\n");
    // }


}
