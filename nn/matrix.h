#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>

#ifndef MATRIX_H
#define MATRIX_H






typedef struct 
{
    size_t row;
    size_t col;
    size_t stride; // co
    float* mat;
}Matrix;


Matrix* allocateMatrix(const size_t row, const size_t col)
{
    //srand(time(NULL));
    Matrix* matrix = (Matrix*) malloc(sizeof(Matrix));
    matrix->row = row;
    matrix->col = col;

    matrix->stride = matrix->col;
    matrix->mat = (float*) malloc(sizeof(float)*col*row);

    return matrix;

}

float getElementAt(const Matrix *matrix, int row, int col)
{
    return matrix->mat[(matrix->col)*row + col];
}

void setElementAt(Matrix *matrix, int row, int col, float value)
{

    matrix->mat[(matrix->col)*row + col] = value;
}

void printMatrix(const Matrix *matrix)
{

    for(size_t i = 0 ; i<matrix->row ; i++)
    {
        printf("[");
        for(size_t j=0; j<matrix->col; j++ )
        {
            printf("%f ",getElementAt(matrix,i,j));
        }
        printf("]\n");
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

Matrix* matrixMultiply(const Matrix *A, const Matrix *B)
{
    
    assert(A->col == B->row );
    Matrix *prod;
    prod = allocateMatrix(A->row, B->col);
    
    for(size_t i = 0 ; i<prod->row; i++)
    {
        for(size_t j=0; j<prod->col; j++)
        {
            float temp=0;
            for(size_t k=0; k<A->col; k++)
            {
                temp += getElementAt(A,i,k)*getElementAt(B,k,j); 
                
            }
            setElementAt(prod,i,j,temp);
        }
    }
    return prod;
}

Matrix* transposeMatrix(Matrix *matrix)
{
    Matrix *temp;
    temp = allocateMatrix(matrix->col, matrix->row);
    
    for(size_t i = 0 ; i < matrix->row; i++)
    {
        for(size_t j = 0; j < matrix->col; j++)
        {
            // Swap the rows and columns during transpose
            setElementAt(temp, j, i, getElementAt(matrix, i, j));
        }
    }
   
   
    return temp;
    
}

Matrix* indentityMatrix(size_t size)
{
    Matrix* matrix = allocateMatrix(size,size);
   
     for(size_t i=0; i< matrix->row; i++)
        for(size_t j=0; j< matrix->col; j++)
            {
                
                if(i==j) setElementAt(matrix, i,j,1);
                else setElementAt(matrix, i,j,0);
            }
    return matrix;
}

Matrix* addMatrix(const Matrix* A, const Matrix* B)
{
    assert(A->col == B->col);
    assert(A->row == B->row);

    Matrix* sum = allocateMatrix(A->row, A->col);
    for(size_t i=0; i<A->row; i++)
    {
        for(size_t j=0; j<A->col; j++)
        {
            setElementAt(sum,i,j,getElementAt(A,i,j)+getElementAt(B,i,j));
        }
    }

    return sum;
}

Matrix* subtractMatrix(const Matrix* A, const Matrix* B)
{
    assert(A->col == B->col);
    assert(A->row == B->row);

    Matrix* sum = allocateMatrix(A->row, A->col);
    for(size_t i=0; i<A->row; i++)
    {
        for(size_t j=0; j<A->col; j++)
        {
            setElementAt(sum,i,j,getElementAt(A,i,j)-getElementAt(B,i,j));
        }
    }

    return sum;
}

float sigmoid(float x)
{
    return (float) 1/(1+exp(-x));
}

float sigmoid_derivative(float x)
{
    return sigmoid(x)*(1-sigmoid(x));
}

Matrix* broadcastFunction(float (*function)(float), Matrix* A)
{
    Matrix *new = allocateMatrix(A->row,A->col);

    for(size_t i=0; i<A->row; i++)
        for(size_t j=0; j<A->col; j++)
        {
            setElementAt(new,i,j,function(getElementAt(A,i,j)));
        }

    return new;
}

Matrix* hadamardProduct(const Matrix *A, const Matrix *B)
{
    
    assert(A->col == B->col && A->row == B->row);
    Matrix *prod;
    prod = allocateMatrix(A->row, A->col);
    
    for(size_t i = 0; i< prod->row; i++)
    {
        for(size_t j=0; j< prod->col; j++)
        {
           
            setElementAt(prod,i,j,getElementAt(A,i,j)*getElementAt(B,i,j));
        }
    }
    return prod;
}

void copyMatrix(Matrix* destination, const Matrix* source) {
    assert (source->row == destination->row && source->col == destination->col);
        // Handle the case where source and destination matrices have different dimensions.
        // You may choose to return an error code or handle it in another way.
  

    for (size_t i = 0; i < source->row; i++) {
        for (size_t j = 0; j < source->col; j++) {
            float value = getElementAt(source, i, j);
            setElementAt(destination, i, j, value);
        }
    }
}

Matrix* scaleMatrix(const Matrix* A, float scalar) {
    Matrix* scaled = allocateMatrix(A->row, A->col);

    for (size_t i = 0; i < A->row; i++) {
        for (size_t j = 0; j < A->col; j++) {
            float value = getElementAt(A, i, j) * scalar;
            setElementAt(scaled, i, j, value);
        }
    }

    return scaled;
}

float relu(float x) {
    if (x > 0) {
        return x;
    } else {
        return 0.0;
    }
}

float relu_derivative(float x) {
    if (x > 0) {
        return 1.0;
    } else {
        return 0.0;
    }
}


float linear(float x)
{
    return x;
}

float linear_derivative(float x)
{
    return (float) 1 ;
}


#endif