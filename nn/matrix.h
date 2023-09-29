#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>



typedef struct 
{
    size_t row;
    size_t col;
    size_t stride; // co
    float* mat;
}Matrix;


Matrix* allocateMatrix(size_t row, size_t col)
{
    srand(time(NULL));
    Matrix* matrix = (Matrix*) malloc(sizeof(Matrix));
    matrix->row = row;
    matrix->col = col;

    int size = matrix->col * matrix->row;
    matrix->stride = size/matrix->col;
    matrix->mat = (float*) malloc(sizeof(float)*col*row);

    return matrix;

}

float getElementAt(Matrix *matrix, int row, int col)
{
    return matrix->mat[2*row + col];
}

void setElementAt(Matrix *matrix, int row, int col, float value)
{
    matrix->mat[2*row + col] = value;
}

void printMatrix(Matrix *matrix)
{

    for(int i = 0 ; i<matrix->stride ; i++)
    {
        for(int j=0; j<matrix->col; j++ )
        {
            printf("%f ",getElementAt(matrix,i,j));
        }
        printf("\n");
    }
}

void randomizeMatrix(Matrix *matrix)
{
    
    for(size_t i=0; i<matrix->row; i++)
    {
        for(size_t j=0; j<matrix->col; j++)
        {
            setElementAt(matrix,i,j, (float) rand()/(RAND_MAX));
        }
    }
}

Matrix* matrixMultiply(Matrix *A, Matrix *B)
{
    float temp=0;
    assert(A->col == B->row );
    Matrix *prod;
    prod = allocateMatrix(A->row, B->col);
    
    for(size_t i = 0 ; i<prod->row; i++)
    {
        for(size_t j=0; j<prod->col; j++)
        {
            for(size_t k=0; k<A->col; k++)
            {
                temp += getElementAt(A,i,k)*getElementAt(B,k,j); 
                setElementAt(prod,i,j,temp);
            }
        }
    }
    return prod;
}