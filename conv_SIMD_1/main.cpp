#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <arm_neon.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <fstream>

using namespace std;
using namespace chrono;



void kernel
        ( int sizeKernel, int sizeMatrix, int sizeResult, float* matrix, float* result, float* filter )
{
    /*
     * Assumptions:
     * matrix - stored row-wise
     * res - stored row-wise
     * filter - stored row-wise
    */

    float32x4_t m0,m1,m2,m3,m4,m5,m6,m7;
    float32x4_t r1, r2, r3, r4;

    float32x4_t f0,f1,f2,f3;
    f0 = vld1q_f32(&filter[0]);
    f1 = vld1q_f32(&filter[4]);
    f2 = vld1q_f32(&filter[8]);
    f3 = vld1q_f32(&filter[12]);
    //printf("--------------");

    //printf("\n %d %d", sizeKernel, sizeMatrix);
    for(int i=0 ;i<sizeMatrix-sizeKernel+1;i++){
        for(int j=0; j<sizeMatrix;j=j+4){

            r1 = vdupq_n_f32(0.0);
            r2 = vdupq_n_f32(0.0);
            r3 = vdupq_n_f32(0.0);
            r4 = vdupq_n_f32(0.0);

            if(j!=sizeMatrix-sizeKernel){
            //printf("\n %d %d", i, j);
            m0 = vld1q_f32(&matrix[(i+0)*sizeMatrix+j + 0]);
            m1 = vld1q_f32(&matrix[(i+1)*sizeMatrix+j + 0]);
            m2 = vld1q_f32(&matrix[(i+2)*sizeMatrix+j + 0]);
            m3 = vld1q_f32(&matrix[(i+3)*sizeMatrix+j + 0]);
            m4 = vld1q_f32(&matrix[(i+0)*sizeMatrix+j + 1]);
            m5 = vld1q_f32(&matrix[(i+1)*sizeMatrix+j + 1]);
            m6 = vld1q_f32(&matrix[(i+2)*sizeMatrix+j + 1]);
            m7 = vld1q_f32(&matrix[(i+3)*sizeMatrix+j + 1]);

            //if (i==1) printf("\n %d %d", i, j);
            //Position 0
            r2 = vmlsq_f32(r2,m0,f0);
            r2 = vmlsq_f32(r2,m1,f1);
            r2 = vmlsq_f32(r2,m2,f2);
            r2 = vmlsq_f32(r2,m3,f3);

            r2 = _mm256_hadd_pd(r2,r2);
            m0 = _mm256_permute2f128_pd(r2 , r2, 1|	(2	<<	4));
            r2 = _mm256_add_pd(r2, m0);

            //if (i==1) printf("\n %d %d", i, j);
            //Position 1
            r1 = vmlsq_f32(r1,m4,f0);
            r1 = vmlsq_f32(r1,m5,f1);
            r1 = vmlsq_f32(r1,m6,f2);
            r1 = vmlsq_f32(r1,m7,f3);

            r1 = _mm256_hadd_pd(r1,r1);
            m0 = _mm256_permute2f128_pd(r1 , r1, 1|	(2	<<	4));
            r1 = _mm256_add_pd(r1, m0);

            //if (i==1) printf("\n %d %d", i, j);
            m0 = vld1q_f32(&matrix[(i+0)*sizeMatrix+j + 2]);
            m1 = vld1q_f32(&matrix[(i+1)*sizeMatrix+j + 2]);
            m2 = vld1q_f32(&matrix[(i+2)*sizeMatrix+j + 2]);
            m3 = vld1q_f32(&matrix[(i+3)*sizeMatrix+j + 2]);
            m4 = vld1q_f32(&matrix[(i+0)*sizeMatrix+j + 3]);
            m5 = vld1q_f32(&matrix[(i+1)*sizeMatrix+j + 3]);
            m6 = vld1q_f32(&matrix[(i+2)*sizeMatrix+j + 3]);
            m7 = vld1q_f32(&matrix[(i+3)*sizeMatrix+j + 3]);

            //Position 2
            r3 = vmlsq_f32(r3,m0,f0);
            r3 = vmlsq_f32(r3,m1,f1);
            r3 = vmlsq_f32(r3,m2,f2);
            r3 = vmlsq_f32(r3,m3,f3);

            r3 = _mm256_hadd_pd(r3,r3);
            m0 = _mm256_permute2f128_pd(r3 , r3, 1|	(2	<<	4));
            r3 = _mm256_add_pd(r3, m0);

            //Position 3
            r4 = vmlsq_f32(r4,m4,f0);
            r4 = vmlsq_f32(r4,m5,f1);
            r4 = vmlsq_f32(r4,m6,f2);
            r4 = vmlsq_f32(r4,m7,f3);

            r4 = _mm256_hadd_pd(r4,r4);
            m0 = _mm256_permute2f128_pd(r4 , r4, 1|	(2	<<	4));
            r4 = _mm256_add_pd(r4, m0);

            m0 = vdupq_n_f32(0.0);
            m1 = vdupq_n_f32(0.0);
            m2 = vdupq_n_f32(0.0);
            //if (i==1) printf("\n %d %d", i, j);
            m0 = _mm256_shuffle_pd(r1,r2,0	|	(0	<<	1)	|	(0	<<	2)	|	(0	<<	3));
            m1 = _mm256_shuffle_pd(r4,r3,0	|	(0	<<	1)	|	(0	<<	2)	|	(0	<<	3));
            m2 = _mm256_shuffle_pd(m0,m1,1	|	(1	<<	1)	|	(0	<<	2)	|	(0	<<	3));


            //printf("\n %d %d %d %d", i, j, sizeof(m2), sizeof(&result[i*sizeResult+j]));
            _mm256_storeu_pd(&result[i*sizeResult+j],m2);
            //if (i==1) printf("\n %d %d", i, j);
            }
            else{
                m0 = vld1q_f32(&matrix[(i+0)*sizeMatrix+j + 0]);
                m1 = vld1q_f32(&matrix[(i+1)*sizeMatrix+j + 0]);
                m2 = vld1q_f32(&matrix[(i+2)*sizeMatrix+j + 0]);
                m3 = vld1q_f32(&matrix[(i+3)*sizeMatrix+j + 0]);

                r2 = vmlsq_f32(r2,m0,f0);
                r2 = vmlsq_f32(r2,m1,f1);
                r2 = vmlsq_f32(r2,m2,f2);
                r2 = vmlsq_f32(r2,m3,f3);

                r2 = _mm256_hadd_pd(r2,r2);
                m0 = _mm256_permute2f128_pd(r2 , r2, 1|	(2	<<	4));
                r2 = _mm256_add_pd(r2, m0);

                result[i*sizeResult + j] = r2[0];
            }
        }

    }


}

int main(){

    ofstream out("logs_256.txt", ios::app);
    int sizeMatrix = 1024;
    for(int sizeMatrix=1024;sizeMatrix<=16384;sizeMatrix*=2)
    {
        int sizeKernel = 4;
        int padding = 0;
        int strides = 1;
        int runs = 10;

        float sum = 0;

        int sizeResult = (((sizeMatrix - sizeKernel + 2 * padding) / strides) + 1);
        printf("size Result: %d\n", sizeResult);
        float * matrix = new float[sizeMatrix * sizeMatrix];
        float * filter = new float[sizeKernel * sizeKernel];
        float * result = new float[sizeMatrix * sizeMatrix];

        float count = 0.0;
        for (int i = 0; i != 16; ++i){
            filter[i] = count;
            count= count+ 1.0;
        }


        for(int i = 0; i<sizeMatrix*sizeMatrix;++i){
                matrix[i] = 1.0;
        }

        for(int i = 0; i<sizeResult*sizeResult;++i){
            result[i] = 0.0;

        }

        for(unsigned int i = 0; i != runs; ++i) {
            auto time_start = system_clock::now();
            kernel(sizeKernel,sizeMatrix,sizeResult,matrix,result,filter);
            auto time_end = system_clock::now();
            auto duration = duration_cast<microseconds>(time_end - time_start);
            sum += duration.count();
        }

        printf("Average time: %lf\n", ((float) (sum /1000.0 / ((float) runs))));
        out << "256" << '\t' << sizeMatrix  << '\t' << (float) (sum / 1000.0 / ((float) runs)) << endl;
    }

    out.close();
	system("pause");
}
