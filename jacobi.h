#ifndef JACOBI_H
#define JACOBI_H

/**
 * @brief   Performs Jacobi's method for solving A*x=b for x.
 *
 * @param n                 The size of the input.
 * @param A                 The input matrix `A` of size n-by-n.
 * @param b                 The input vector `b` of size n.
 * @param x                 The output vector `x` of size n.
 * @param max_iter          The maximum number of iterations to run.
 * @param eps			    The termination criteria for the L2-norm of
 *                          ||Ax - b||. Terminates as soon as the total L2-norm
 *                          is smaller or equal to this.
 */
void jacobiSolve(int n, double** A, double* b, double* x, double eps = 1e-10, int maxit = 100);

#endif // JACOBI_H
