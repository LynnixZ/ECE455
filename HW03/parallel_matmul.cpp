// parallel_matmul.cpp
#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    const int N = 512; // 512x512
    std::vector<int> A(N * N, 1);
    std::vector<int> B(N * N, 2);
    std::vector<int> C(N * N, 0);
    std::vector<int> BT(N * N, 0); // transpose of B

    // Transpose B -> BT to improve memory locality
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            BT[j * N + i] = B[i * N + j];

    // Parallel matmul using BT
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int s = 0;
            const int *ai = &A[i * N];
            const int *btj = &BT[j * N];
            for (int k = 0; k < N; ++k) {
                s += ai[k] * btj[k];
            }
            C[i * N + j] = s;
        }
    }

    std::cout << "C[0][0] = " << C[0] << std::endl;
    return 0;
}
