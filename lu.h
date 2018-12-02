#ifndef LU_H
#define LU_H

/**
 * @brief   Performs LU Decomposition method for solving A*x=b for x.
 *
 * @param n                 The size of the input.
 * @param A                 The input matrix `A` of size n-by-n.
 * @param b                 The input vector `b` of size n.
 * @param x                 The output vector `x` of size n.
 */
void LUSolve(int n, double** A, double* b, double *x, double eps=1e-10);

#endif //LU_H
