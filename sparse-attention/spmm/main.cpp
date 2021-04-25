//
// Created by   Hongqian Yu on 24/4/2021.
//

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include "cusprase_impl.h"
#include "cycleTimer.h"
#define MAX_SEQ_LEN 512
#define MAX_BATCH_SIZE 32

void usage(const char* progname) {
    printf("Usage: %s [options] scenename\n", progname);
    printf("Valid scenenames are: rgb, rgby, rand10k, rand100k, biglittle, littlebig, pattern, bouncingballs, fireworks, hypnosis, snow, snowsingle\n");
    printf("Program Options:\n");
    printf("  -c  --check                Check correctness of output\n");
    printf("  -f  --file  <FILENAME>     Dump frames in benchmark mode (FILENAME_xxxx.ppm)\n");
    printf("  -r  --renderer <ref/cuda>  Select renderer: ref or cuda\n");
    printf("  -s  --size  <INT>          Size of the matrix <INT>x<INT> \n");
    printf("  -?  --help                 This message\n");
}

void print_arr(float *arr, int n) {
    for(int i = 0; i <  n; i++) {
        printf("%f ", arr[i]);
    }
}

// TODO: generate csr format sparse input
void get_matrix_from_stdin(float **matrix) {

}

int get_idx(int row, int col, int width) {
    return row * width + col;
}

// res(m, n) = sparse(m, k) * dense(k, n)
int main(int argc, char** argv)
{


    int i, j, m, k, n;
    char c;
    scanf("%d %d", &m, &k);
    float *h_A_dense = (float *)malloc(m * k * sizeof(float));
    for(i = 0; i < m; i++) {
        for(j = 0; j < k; j++) {
            scanf("%f%c", &h_A_dense[get_idx(i, j, k)], &c);
        }
    }

    scanf("%d %d", &k, &n);    // width_sparse == height_dense
    float *h_B_dense = (float *)malloc(k * n * sizeof(float));
    for(i = 0; i < k; i++) {
        for(j = 0; j < n; j++) {
            scanf("%f%c", &h_B_dense[get_idx(i, j, n)], &c);
        }
    }

    double start = CycleTimer::currentSeconds();
    cusparse_mmul(h_A_dense, h_B_dense, m, k, n);
    double end = CycleTimer::currentSeconds();
    printf("cusparse:    %.4f ms\n", 1000.f * (end - start));

    return 0;
}
