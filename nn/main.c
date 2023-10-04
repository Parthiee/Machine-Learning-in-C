#include <stdio.h>
#include "matrix.h"

int main()
{
    Matrix *A, *C;
    A = allocateMatrix(3,3);
    C = allocateMatrix(3,3);

    
    // printMatrix(A);
    // printMatrix(B);

   // Matrix *C =  matrixMultiply(A,B);
    printMatrix(C);

    Matrix *t = transposeMatrix(C);
    printf("\n\n");
    printMatrix(t);

    return 0;
}