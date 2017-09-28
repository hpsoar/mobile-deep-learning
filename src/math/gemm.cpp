/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/
#include "math/gemm.h"

namespace mdl {
    vector<Gemmer *> Gemmer::gemmers;

    void Gemmer::pack_MRxk(int k, const float *A, int incRowA, int incColA, float *buffer) {
        int j, a2 = incRowA, a3 = 2 * incRowA, a4 = 3 * incRowA;
        for (j = 0; j < k; ++j) {
            // for (int i = 0; i < MR; ++i) {
            //     buffer[i] = A[i * incRowA];
            // }
            buffer[0] = A[0];
            buffer[1] = A[a2];
            buffer[2] = A[a3];
            buffer[3] = A[a4];
            A += 1; // move to next col
            buffer += MR; // move to next MR
        }
    }
    
    /*
     * mc x kc, split to [MR x kc], [MR x kc], ..., put this blocks into buffer in a squential way
     * eg. mc = 8, kc = 4, each block [4 x 4]
     * [1 2 3 4]
     * [5 6 7 8]
     * [9 10 11 12]
     * [13 14 15 16],
     * [17 18 19 20]
     * [21 22 23 24],
     * [25 26 27 28],
     * [29 30 31 32]
     * now buffer:
     * [1 5 9 13 2 6 10 14 3 7 11 15 4 8 12 16]
     * [17 21 25 29 18 22 26 30 19 23 27 31 20 24 28 32]
     
     * [1 2 3 4 9 10 11 12 17 18 19 20 25 26 27 28]
     * [5 6 7 8 13 14 15 16 21 22 23 24 29 30 31 32]
     
     * AB = [1*1 5*2 9*3 13*4 2*9 6*10 10*11 14*12 3*17 7*18 11*19 15*20 4*25 8*26 12*27 16*28]
     * C = [1*1 2*9 3*17 4*25],
     *     [5*2 6*10 7*18 8*26],
     *     [9*3 10*11 11*19 12*27],
     *     [13*4 14*12 15*20 16*28]
     */
    void Gemmer::pack_A(int mc, int kc, const float *A, int incRowA, int incColA, float *buffer) {
        int mp = mc / MR;
        int _mr = mc % MR;
        int tmp1 = kc * MR;
        int tmp2 = MR * incRowA;
        int i, j;
    
        for (i = 0; i < mp; ++i) {
            pack_MRxk(kc, A, incRowA, incColA, buffer);
            buffer += tmp1; // move a block
            A += tmp2; // move MR row
            // buffer += kc * MR;
            // A += MR * incRowA;
        }
        if (_mr > 0) {
            for (j = 0; j < kc; ++j) {
                for (i = 0; i < _mr; ++i) {
                    buffer[i] = A[i * incRowA];
                }
                for (i = _mr; i < MR; ++i) {
                    buffer[i] = 0.0;
                }
                A += 1;
                buffer += MR;
            }
        }
    }
    
    void Gemmer::pack_kxNR(int k, const float *B, int incRowB, int incColB, float *buffer) {
        int i, j;
        for (i = 0; i < k; ++i) {
            for (j = 0; j < NR; ++j) {
                buffer[j] = B[j];
            }
            // float32x4_t bv = vld1q_f32(B);
            // vst1q_f32(buffer, bv);
            B += incRowB; // move a row
            buffer += NR; // move to next NR
        }
    }
    
    /*
     * kc x nc, split to [kc x NR], [kc x NR], ..., put this blocks into buffer in a squential way
     * eg. kc = 4, nc = 8
     * [1 2 3 4 5 6 7 8]
     * [9 10 11 12 13 14 15 16],
     * [17 18 19 20 21 22 23 24],
     * [25 26 27 28 29 30 31 32]
     * now buffer:
     * [1 2 3 4 9 10 11 12 17 18 19 20 25 26 27 28]
     * [5 6 7 8 13 14 15 16 21 22 23 24 29 30 31 32]
     */
    void Gemmer::pack_B(int kc, int nc, const float *B, int incRowB, int incColB, float *buffer) {
        int np = nc / NR;
        int _nr = nc % NR;
        int tmp1 = kc * NR;
        int i, j;
    
        for (j = 0; j < np; ++j) {
            pack_kxNR(kc, B, incRowB, incColB, buffer);
            B += NR; // move NR column
            buffer += tmp1; // move a block
        }
        if (_nr > 0) {
            for (i = 0; i < kc; ++i) {
                for (j = 0; j < _nr; ++j) {
                    buffer[j] = B[j];
                }
                for (j = _nr; j < NR; ++j) {
                    buffer[j] = 0.0;
                }
                buffer += NR;
                B += incRowB;
            }
        }
    }
    
    /*
     * output: [MR x kc] * [kc x NR] = [MR x NR]
     */
    void Gemmer::dgemm_micro_kernel(int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC, int incColC) {
#ifndef MDL_MAC
        int i, j, l;
        float32x4_t abv0 = vdupq_n_f32(0);
        float32x4_t abv1 = vdupq_n_f32(0);
        float32x4_t abv2 = vdupq_n_f32(0);
        float32x4_t abv3 = vdupq_n_f32(0);

        float32x4_t av;
        float32x4_t bv;

        float32x2_t bv01;
        float32x2_t bv23;

        for (l = 0; l < kc; ++l) {
            av = vld1q_f32(A);
            bv = vld1q_f32(B);
            bv01 = vget_low_f32(bv);
            abv0 = vmlaq_lane_f32(abv0, av, bv01, 0);
            abv1 = vmlaq_lane_f32(abv1, av, bv01, 1);
            bv23 = vget_high_f32(bv);
            abv2 = vmlaq_lane_f32(abv2, av, bv23, 0);
            abv3 = vmlaq_lane_f32(abv3, av, bv23, 1);
            A += MR;
            B += NR;
        }

        vst1q_f32(AB_ + 0, abv0);
        vst1q_f32(AB_ + 4, abv1);
        vst1q_f32(AB_ + 8, abv2);
        vst1q_f32(AB_ + 12, abv3);
#else
        int i = 0;
        int j = 0;
        int l = 0;
        for (l = 0; l < MR * NR; ++l) {
            AB_[l] = 0;
        }
        for (l = 0; l < kc; ++l) {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    AB_[i + j * MR] += A[i] * B[j];
                }
            }
            A += MR;
            B += NR;
        }
#endif
        // initialize C, with beta
    if (equal(beta, 0.0)){
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = 0.0;
                }
            }
        } else if (!equal(beta,1.0)) {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] *= beta;
                }
            }
        }
    
        if (!equal(alpha, 1.0)) {
            for (j = 0; j < NR; ++j) { // C: next col, AB_: next row
                for (i = 0; i < MR; ++i) { // C: next row, AB_: next col
                    // AB_:（行序） 0, 1, 2...15 --> C:（usually 列序, if incRowC & incColC is interchanged, it will be 行序）
                    C[i * incRowC + j * incColC] += alpha * AB_[i + j * MR];
                }
            }
        } else {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] += AB_[i + j * MR];
                }
            }
        }
    }
    
    void Gemmer::dgeaxpy(int m, int n, float alpha, const float *X, int incRowX, int incColX, float *Y, int incRowY, int incColY) {
        int i, j;
        if (!equal(alpha, 1.0)) {
            for (j = 0; j < n; ++j) {
                for (i = 0; i < m; ++i) {
                    Y[i * incRowY + j] += alpha * X[i + j * incColX];
                }
            }
        } else {
            for (j = 0; j < n; ++j) { // X: next row, Y: next col
                for (i = 0; i < m; ++i) { // X: next col, Y: next row
                    Y[i * incRowY + j] += X[i + j * incColX];
                }
            }
        }
    }
    
    void Gemmer::dgescal(int m, int n, float alpha, float *X, int incRowX, int incColX) {
        int i, j;
        if (!equal(alpha, 0.0)) {
            for (i = 0; i < m; ++i) {
                for (j = 0; j < n; ++j) {
                    X[i * incRowX + j] *= alpha;
                }
            }
        } else {
            for (i = 0; i < m; ++i) {
                for (j = 0; j < n; ++j) {
                    X[i * incRowX + j] = 0.0;
                }
            }
        }
    }
    
    /*
     * 64 x 147 * 147 * 4096
     */
    void Gemmer::dgemm_macro_kernel(int mc, int nc, int kc, float alpha, float beta, float *C, int incRowC, int incColC) {
        int mp = (mc + MR - 1) / MR; // 16
        int np = (nc + NR - 1) / NR; // 1024
    
        int _mr = mc % MR;
        int _nr = nc % NR;
    
        int i, j;
    
        for (j = 0; j < np; ++j) {
            int nr = (j != np - 1 || _nr == 0) ? NR : _nr;
    
            for (i = 0; i < mp; ++i) {
                int mr = (i != mp - 1 || _mr == 0) ? MR : _mr;
    
                if (mr == MR && nr == NR) {
                    dgemm_micro_kernel(kc, alpha, &A_[i * kc * MR], &B_[j * kc * NR], beta, &C[i * MR * incRowC + j * NR], incRowC, incColC);
                } else {
                    // last columns or rows
                    
                    // calc A_ * B_ -> C_, compute all [MR x NR], and store to C_ in row order, may be more efficient
                    dgemm_micro_kernel(kc, alpha, &A_[i * kc * MR], &B_[j * kc * NR], 0.0, C_, 1, MR);
                    
                    // then fetch & store to C, only those needed positions
                    
                    // scale C with beta
                    dgescal(mr, nr, beta, &C[i * MR * incRowC + j * NR], incRowC, incColC);
                    
                    // accumulate C_ to C
                    dgeaxpy(mr, nr, 1.0, C_, 1, MR, &C[i * MR * incRowC + j * NR], incRowC, incColC);
                }
            }
        }
    }
    
    /*
     * split A & B to blocks [MC x KC], [KC x NC], then multiple & accumulate blocks using dgemm_macro_kernel
     */
    void Gemmer::dgemm_nn(int m, int n, int k, float alpha, const float *A, int incRowA, int incColA, const float *B, int incRowB, int incColB, float beta, float *C, int incRowC, int incColC) {
        int mb = (m + MC - 1) / MC;
        int nb = (n + NC - 1) / NC;
        int kb = (k + KC - 1) / KC;
    
        int _mc = m % MC;
        int _nc = n % NC;
        int _kc = k % KC;
    
        int mc, nc, kc;
        int i, j, l;
    
        float _beta;
    
        if (equal(alpha, 0.0) ||  k == 0) {
            dgescal(m, n, beta, C, incRowC, incColC);
            return;
        }
    
        for (j = 0; j < nb; ++j) {
            nc = (j != nb - 1 || _nc == 0) ? NC : _nc;
    
            for (l = 0; l < kb; ++l) {
                kc = (l != kb - 1 || _kc == 0) ? KC : _kc;
                
                // if we split in k-dim, the first iteration use beta, following will use 1.0 because the following should be accumulated, each block only computes partial of the output
                _beta = (l == 0) ? beta : 1.0;
    
                // pack a block of size [kc x nc] in B to B_, eg. [147 x 4096]
                pack_B(kc, nc, &B[l * KC * incRowB + j * NC], incRowB, incColB, B_);
    
                for (i = 0; i < mb; ++i) {
                    mc = (i != mb - 1 || _mc == 0) ? MC : _mc;
    
                    // pack a block of size [mc x kc] in A to A_, eg. [64 * 147]
                    pack_A(mc, kc, &A[i * MC * incRowA + l * KC], incRowA, incColA, A_);
    
                    // 64 x 147 * 147 * 4096
                    dgemm_macro_kernel(mc, nc, kc, alpha, _beta, &C[i * MC * incRowC + j * NC], incRowC, incColC);
                }
            }
        }
    }

    void Gemmer::sgemm(int m, int n, int k, const float *A, const float *B, float *C) {
        dgemm_nn(m, n, k, 1, A, k, 1, B, n, 1, 0, C, n, 1); // alpha == 1, beta == 0
    }

    void Gemmer::sgemm(int m, int n, int k, const float *A, const float *B, float *C, float alpha, float beta) {
        dgemm_nn(m, n, k, alpha, A, k, 1, B, n, 1, beta, C, n, 1);
    }
};
