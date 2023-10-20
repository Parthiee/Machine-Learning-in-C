# Matrix Library Documentation

This library provides a set of matrix manipulation functions, including matrix creation, element access, mathematical operations, and more. Below is a brief documentation for the functions and structures in this library.

# Matrix Structure

```c
typedef struct {
    size_t row;
    size_t col;
    float* mat;
} Matrix;
```
-   `row`: The number of rows in the matrix.
-   `col`: The number of columns in the matrix.
-   `mat`: A pointer to the matrix data stored as a one-dimensional array.


## Function Documentation

### `Matrix* allocateMatrix(const size_t row, const size_t col)`

Allocates memory for a new matrix with the specified number of rows and columns.

### `float getElementAt(const Matrix *matrix, int row, int col)`

Returns the value of an element at a specific row and column in the matrix.

### `void setElementAt(Matrix *matrix, int row, int col, float value)`

Sets the value of an element at a specific row and column in the matrix.

### `void printMatrix(const Matrix *matrix)`

Prints the elements of a matrix to the standard output.

### `void randomizeMatrix(Matrix *matrix)`

Fills the matrix with random float values between 0 and 1.

### `Matrix* matrixMultiply(const Matrix *A, the Matrix *B)`

Performs matrix multiplication between two matrices `A` and `B` and returns the result as a new matrix.

### `Matrix* transposeMatrix(Matrix *matrix)`

Transposes a matrix (swaps rows and columns) and returns the transposed matrix as a new one.

### `Matrix* identityMatrix(size_t size)`

Creates an identity matrix of the specified size and returns it.

### `Matrix* addMatrix(const Matrix* A, const Matrix* B)`

Adds two matrices `A` and `B` element-wise and returns the result as a new matrix.

### `Matrix* subtractMatrix(const Matrix* A, const Matrix* B)`

Subtracts matrix `B` from matrix `A` element-wise and returns the result as a new matrix.

### `float sigmoid(float x)`

Calculates the sigmoid function on a single float value.

### `float sigmoid_derivative(float x)`

Calculates the derivative of the sigmoid function for a single float value.

### `Matrix* broadcastFunction(float (*function)(float), Matrix* A)`

Applies a specified function element-wise to the elements of a matrix `A` and returns the result as a new matrix.

### `Matrix* hadamardProduct(const Matrix *A, const Matrix *B)`

Computes the Hadamard product (element-wise multiplication) of two matrices `A` and `B` and returns the result as a new matrix.

### `void copyMatrix(Matrix* destination, const Matrix* source)`

Copies the contents of a source matrix to a destination matrix. Both matrices should have the same dimensions.

### `Matrix* scaleMatrix(const Matrix* A, float scalar)`

Scales a matrix `A` by a scalar value and returns the scaled matrix as a new one.

### `float relu(float x)`

Calculates the rectified linear unit (ReLU) function on a single float value.

### `float relu_derivative(float x)`

Calculates the derivative of the ReLU function for a single float value.

### `float linear(float x)`

Returns the input float value as is (identity function).

### `float linear_derivative(float x)`

Returns a constant value of 1 as the derivative of the linear function for a single float value.

This library provides a set of basic matrix operations and mathematical functions that can be used for various numerical and machine learning tasks.## Function Documentation

### `Matrix* allocateMatrix(const size_t row, const size_t col)`

Allocates memory for a new matrix with the specified number of rows and columns.

### `float getElementAt(const Matrix *matrix, int row, int col)`

Returns the value of an element at a specific row and column in the matrix.

### `void setElementAt(Matrix *matrix, int row, int col, float value)`

Sets the value of an element at a specific row and column in the matrix.

### `void printMatrix(const Matrix *matrix)`

Prints the elements of a matrix to the standard output.

### `void randomizeMatrix(Matrix *matrix)`

Fills the matrix with random float values between 0 and 1.

### `Matrix* matrixMultiply(const Matrix *A, the Matrix *B)`

Performs matrix multiplication between two matrices `A` and `B` and returns the result as a new matrix.

### `Matrix* transposeMatrix(Matrix *matrix)`

Transposes a matrix (swaps rows and columns) and returns the transposed matrix as a new one.

### `Matrix* identityMatrix(size_t size)`

Creates an identity matrix of the specified size and returns it.

### `Matrix* addMatrix(const Matrix* A, const Matrix* B)`

Adds two matrices `A` and `B` element-wise and returns the result as a new matrix.

### `Matrix* subtractMatrix(const Matrix* A, const Matrix* B)`

Subtracts matrix `B` from matrix `A` element-wise and returns the result as a new matrix.

### `float sigmoid(float x)`

Calculates the sigmoid function on a single float value.

### `float sigmoid_derivative(float x)`

Calculates the derivative of the sigmoid function for a single float value.

### `Matrix* broadcastFunction(float (*function)(float), Matrix* A)`

Applies a specified function element-wise to the elements of a matrix `A` and returns the result as a new matrix.

### `Matrix* hadamardProduct(const Matrix *A, const Matrix *B)`

Computes the Hadamard product (element-wise multiplication) of two matrices `A` and `B` and returns the result as a new matrix.

### `void copyMatrix(Matrix* destination, const Matrix* source)`

Copies the contents of a source matrix to a destination matrix. Both matrices should have the same dimensions.

### `Matrix* scaleMatrix(const Matrix* A, float scalar)`

Scales a matrix `A` by a scalar value and returns the scaled matrix as a new one.

### `float relu(float x)`

Calculates the rectified linear unit (ReLU) function on a single float value.

### `float relu_derivative(float x)`

Calculates the derivative of the ReLU function for a single float value.

### `float linear(float x)`

Returns the input float value as is (identity function).

### `float linear_derivative(float x)`

Returns a constant value of 1 as the derivative of the linear function for a single float value.

This library provides a set of basic matrix operations and mathematical functions that can be used for various numerical and machine learning tasks.
