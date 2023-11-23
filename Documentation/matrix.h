/**
 * @file matrix.h
 * @brief Matrix Operations Header File
 *
 * This header file defines structures and functions for matrix operations, including allocation, manipulation, and mathematical operations.
 * It also includes functions for common activation functions used in neural networks.
 *
 * @author Parthiban
 * @date 23 November 2023
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>

/**
 * @brief Represents a matrix with its dimensions and elements.
 */
typedef struct 
{
    size_t row;     /**< Number of rows in the matrix. */
    size_t col;     /**< Number of columns in the matrix. */
    float* mat;     /**< Array storing matrix elements. */
} Matrix;

/**
 * @brief Allocates memory for a matrix with the specified number of rows and columns.
 *
 * @param row Number of rows in the matrix.
 * @param col Number of columns in the matrix.
 * @return Pointer to the allocated matrix.
 */
Matrix* allocateMatrix(const size_t row, const size_t col);

/**
 * @brief Retrieves the element at the specified row and column in the matrix.
 *
 * @param matrix Pointer to the matrix.
 * @param row Row index of the element.
 * @param col Column index of the element.
 * @return The element at the specified position in the matrix.
 */
float getElementAt(const Matrix *matrix, int row, int col);

/**
 * @brief Sets the element at the specified row and column in the matrix to the given value.
 *
 * @param matrix Pointer to the matrix.
 * @param row Row index of the element.
 * @param col Column index of the element.
 * @param value The value to set at the specified position in the matrix.
 */
void setElementAt(Matrix *matrix, int row, int col, float value);

/**
 * @brief Prints the elements of the matrix.
 *
 * @param matrix Pointer to the matrix to be printed.
 */
void printMatrix(const Matrix *matrix);

/**
 * @brief Initializes the matrix with random values between 0 and 1.
 *
 * @param matrix Pointer to the matrix to be randomized.
 */
void randomizeMatrix(Matrix *matrix);

/**
 * @brief Performs matrix multiplication of two matrices.
 *
 * @param A Pointer to the first matrix.
 * @param B Pointer to the second matrix.
 * @return Pointer to the resulting matrix.
 */
Matrix* matrixMultiply(const Matrix *A, const Matrix *B);

/**
 * @brief Transposes the given matrix.
 *
 * @param matrix Pointer to the matrix to be transposed.
 * @return Pointer to the transposed matrix.
 */
Matrix* transposeMatrix(Matrix *matrix);

/**
 * @brief Creates an identity matrix of the specified size.
 *
 * @param size Size of the identity matrix.
 * @return Pointer to the identity matrix.
 */
Matrix* identityMatrix(size_t size);

/**
 * @brief Adds two matrices element-wise.
 *
 * @param A Pointer to the first matrix.
 * @param B Pointer to the second matrix.
 * @return Pointer to the resulting matrix.
 */
Matrix* addMatrix(const Matrix* A, const Matrix* B);

/**
 * @brief Subtracts one matrix from another element-wise.
 *
 * @param A Pointer to the first matrix.
 * @param B Pointer to the second matrix.
 * @return Pointer to the resulting matrix.
 */
Matrix* subtractMatrix(const Matrix* A, const Matrix* B);

/**
 * @brief Applies the sigmoid activation function to a given value.
 *
 * @param x Input value to the sigmoid function.
 * @return Result of the sigmoid function.
 */
float sigmoid(float x);

/**
 * @brief Calculates the derivative of the sigmoid activation function at a given value.
 *
 * @param x Input value to the sigmoid function.
 * @return Result of the sigmoid derivative function.
 */
float sigmoid_derivative(float x);

/**
 * @brief Broadcasts a given function to each element of a matrix.
 *
 * @param function Pointer to the function to be broadcasted.
 * @param A Pointer to the matrix.
 * @return Pointer to the resulting matrix.
 */
Matrix* broadcastFunction(float (*function)(float), Matrix* A);

/**
 * @brief Performs element-wise multiplication (Hadamard product) of two matrices.
 *
 * @param A Pointer to the first matrix.
 * @param B Pointer to the second matrix.
 * @return Pointer to the resulting matrix.
 */
Matrix* hadamardProduct(const Matrix *A, const Matrix *B);

/**
 * @brief Copies the elements of one matrix to another.
 *
 * @param destination Pointer to the destination matrix.
 * @param source Pointer to the source matrix.
 */
void copyMatrix(Matrix* destination, const Matrix* source);

/**
 * @brief Scales each element of a matrix by a scalar value.
 *
 * @param A Pointer to the matrix.
 * @param scalar Scaling factor.
 * @return Pointer to the resulting scaled matrix.
 */
Matrix* scaleMatrix(const Matrix* A, float scalar);

/**
 * @brief Applies the rectified linear unit (ReLU) activation function to a given value.
 *
 * @param x Input value to the ReLU function.
 * @return Result of the ReLU function.
 */
float relu(float x);

/**
 * @brief Calculates the derivative of the ReLU activation function at a given value.
 *
 * @param x Input value to the ReLU function.
 * @return Result of the ReLU derivative function.
 */
float relu_derivative(float x);

/**
 * @brief Applies the linear activation function to a given value.
 *
 * @param x Input value to the linear function.
 * @return The input value itself.
 */
float linear(float x);

/**
 * @brief Calculates the derivative of the linear activation function at a given value.
 *
 * @param x Input value to the linear function.
 * @return Constant value of 1.
 */
float linear_derivative(float x);

/**
 * @brief Deallocates the memory used by a matrix.
 *
 * @param A Pointer to the matrix to be deallocated.
 */
void killMatrix(Matrix *A);

#endif
