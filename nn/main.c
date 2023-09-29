#include <stdio.h>
#include "LinearAlgebra.h"

int main()
{
    Matrix *A, *B;
    A = allocateMatrix(1,3);
    B = allocateMatrix(3,1);
    randomizeMatrix(A);
    randomizeMatrix(B);
    
    
    // printMatrix(A);
    // printMatrix(B);

    Matrix *C =  matrixMultiply(A,B);
    printMatrix(C);

    return 0;
}