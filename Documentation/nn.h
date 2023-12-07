/**
 * @file nn.h
 * @brief Neural Network Implementation Header File
 *
 * This header file defines structures and functions for implementing a simple feedforward neural network.
 * The neural network consists of layers, and each layer contains weights, biases, and activation functions.
 * Training is performed using backpropagation with gradient descent.
 *
 * @author Parthiban
 * @date 23 November 2023
 */

#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "matrix.h"
#include "glob.h"

/**
 * @brief Represents a single dataset entry.
 *
 * The dataset is an array of dataset entries, where each entry contains input features and the predicted column value.
 */
typedef float dataset[NUM_COLS];

/**
 * @brief Represents a neural network layer.
 *
 * A layer consists of the number of nodes, number of inputs, weight matrix (W), bias matrix (B), and activation matrix (Activations).
 */
typedef struct 
{
    int numNodes;          /**< Number of nodes in the layer */
    int numInputs;         /**< Number of input features to the layer */
    Matrix* W;             /**< Weight matrix for the layer */
    Matrix* B;             /**< Bias matrix for the layer */
    Matrix* Activations;   /**< Activation matrix for the layer */
} Layer;

/**
 * @brief Creates a new layer for the neural network.
 *
 * Initializes a new layer with random weights and biases based on the specified number of nodes and inputs.
 *
 * @param numInputs Number of input features to the layer.
 * @param numNodes Number of nodes in the layer.
 * @return The created layer.
 */
Layer createLayer(int numInputs, int numNodes);

/**
 * @brief Performs forward propagation through the neural network.
 *
 * Given the input features, this function propagates the input forward through the network, applying activation functions.
 *
 * @param layers Array of layers representing the neural network.
 * @param numLayers Number of layers in the neural network.
 * @param ip Input dataset for a single entry.
 * @param activation Activation function for the layers.
 * @param numInputs Number of input features.
 * @return The output matrix after forward propagation.
 */
Matrix* forwardPropagate(Layer* layers, int numLayers, dataset ip, float (*activation)(float), int numInputs);

/**
 * @brief Calculates the cost (error) of the neural network predictions.
 *
 * Computes the mean squared error for the given dataset and neural network predictions.
 *
 * @param layers Array of layers representing the neural network.
 * @param numLayers Number of layers in the neural network.
 * @param data Array of datasets.
 * @param activation Activation function for the layers.
 * @param numInput Number of input features.
 * @return The cost (mean squared error) of the neural network predictions.
 */
float cost(Layer* layers, int numLayers, dataset* data, float (*activation)(float), int numInput);

/**
 * @brief Performs backpropagation to update the neural network weights and biases.
 *
 * Updates the weights and biases of the neural network using backpropagation and gradient descent.
 *
 * @param layers Array of layers representing the neural network.
 * @param numLayers Number of layers in the neural network.
 * @param data Array of datasets.
 * @param activation Activation function for the layers.
 * @param D_activation Derivative of the activation function for backpropagation.
 * @param numInput Number of input features.
 * @return NULL (or a meaningful result based on your requirements).
 */
Matrix* backPropagate(Layer* layers, int numLayers, dataset* data, float (*activation)(float), float (*D_activation)(float), int numInput, int numHidden);

/**
 * @brief Calculates the R-squared score of the neural network predictions.
 *
 * Computes the R-squared score for the given dataset and neural network predictions.
 *
 * @param layers Array of layers representing the neural network.
 * @param data Array of datasets.
 * @param activation Activation function for the layers.
 * @param numInputs Number of input features.
 * @param numLayers Number of layers in the neural network.
 * @param numHidden Number of hidden layers in  the neural network.
 */
void calculateRSquare(Layer *layers, dataset* data, float (*activation)(float), int numInputs, int numLayers);

/**
 * @brief Frees the memory allocated for the neural network layers.
 *
 * Deallocates the memory used by the weight matrices, bias matrices, and activation matrices of each layer.
 *
 * @param layers Array of layers representing the neural network.
 * @param numLayers Number of layers in the neural network.
 */
void freeMemory(Layer* layers, int numLayers);

#endif
