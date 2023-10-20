#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "matrix.h"

#ifndef NN_H
#define NN_H

/*
    AND, NAND works for very large num of neurons

*/
#define PRED_COLUMN 1

#define SIZE_DATA 96  // data

#define L1 1//  INPUT LAYER
#define L2 100// Hidden Layer 1
#define L3 1//  Hidden Layer 2
#define L4 10
#define L5 1 // OUTPUT LAYER
#define INPUT 1
#define EPOCH 100000
#define LEARNING_RATE 0.00001
#define NUM_LAYERS 3

typedef float dataset[2];


typedef struct 
{
    int numNodes;
    int numInputs;
    Matrix* W;
    Matrix* B;
    Matrix* Activations;


}Layer;

Layer createLayer(int numInputs,int numNodes)
{
    Layer l;

    l.numInputs = numInputs;
    l.numNodes = numNodes;
    l.W = allocateMatrix(numInputs, numNodes);
    l.W =transposeMatrix(l.W);
    l.B = allocateMatrix(numNodes, 1);
    l.Activations = allocateMatrix(numNodes, 1);
    randomizeMatrix(l.W);
    randomizeMatrix(l.B);

    return l;
}


Matrix* forwardPropagate(Layer* layers, int numLayers, dataset ip, float (*activation)(float))
{
    Matrix* input = allocateMatrix(INPUT,1);
    Matrix* output;

   
       setElementAt(input,0,0, ip[0]);
       //setElementAt(input,1,0, ip[1]); // Manually change this

       // Y = W*x
       for(int j=0; j<numLayers; j++)
       {
        output = matrixMultiply(layers[j].W,input);
        output = addMatrix(output,layers[j].B);
        output = broadcastFunction(activation,output);
        copyMatrix(layers[j].Activations,output);
        input = output;
       }
       return output;
    

    
}

float cost(Layer* layers, int numLayers, dataset* data, float (*activation)(float))
{
    float cumError=0;
    for(int i=0; i<SIZE_DATA; i++)
    {
        Matrix* output = forwardPropagate(layers,numLayers,data[i], activation);
        float y_hat = getElementAt(output,0,0);
        float y = data[i][PRED_COLUMN];
        cumError += (y - y_hat)*(y - y_hat);
    }

  cumError /= SIZE_DATA;

  return cumError;

}


Matrix* backPropagate(Layer* layers, int numLayers, dataset* data, float (*activation)(float), float (*D_activation)(float)) {
    for (int ep = 0; ep < EPOCH; ep++) {
        if (cost(layers, numLayers, data,activation) < 11.55) {
            break;
        }
        for (int i = 0; i < SIZE_DATA; i++) {
            // Forward propagation
            Matrix* outputs[numLayers];
            Matrix* current_input = allocateMatrix(INPUT, 1);
            setElementAt(current_input, 0, 0, data[i][0]);
            //setElementAt(current_input, 1, 0, data[i][1]);

            outputs[0] = matrixMultiply(layers[0].W, current_input);
            outputs[0] = addMatrix(outputs[0], layers[0].B);
            outputs[0] = broadcastFunction(activation, outputs[0]);
            copyMatrix(layers[0].Activations, outputs[0]);

            for (int l = 1; l < numLayers; l++) {
                outputs[l] = matrixMultiply(layers[l].W, outputs[l - 1]);
                outputs[l] = addMatrix(outputs[l], layers[l].B);
                outputs[l] = broadcastFunction(activation, outputs[l]);
                copyMatrix(layers[l].Activations, outputs[l]);
            }


        if(numLayers == 1)
            {
            

            float y_hat = getElementAt(outputs[numLayers - 1], 0, 0);
            float y = data[i][PRED_COLUMN];
            float output_error = y - y_hat;

            Matrix* delta = allocateMatrix(1, 1);
            setElementAt(delta, 0, 0, output_error);

            Matrix *dw = matrixMultiply(delta, transposeMatrix(current_input));
            layers[0].W = addMatrix(layers[0].W, scaleMatrix(dw, LEARNING_RATE));
            layers[0].B = addMatrix(layers[0].B, scaleMatrix(delta, LEARNING_RATE));


            }

           else{
              // Calculate the error at the output layer
            float y_hat = getElementAt(outputs[numLayers - 1], 0, 0);
            float y = data[i][PRED_COLUMN];
            float output_error = y - y_hat;


   
            // Backpropagation for the last layer (numLayers - 1)
            Matrix* delta = allocateMatrix(1, 1);
            setElementAt(delta, 0, 0, output_error);

            // Update weights and biases for the output layer (numLayers - 1)
            Matrix* dw = matrixMultiply(delta, transposeMatrix(outputs[numLayers - 2]));
            layers[numLayers - 1].W = addMatrix(layers[numLayers - 1].W, scaleMatrix(dw, LEARNING_RATE));
            layers[numLayers - 1].B = addMatrix(layers[numLayers - 1].B, scaleMatrix(delta, LEARNING_RATE));

            // Backpropagation for the intermediate layers
            for (int l = numLayers - 2; l > 0; l--) {
                delta = matrixMultiply(transposeMatrix(layers[l + 1].W), delta);
                delta = hadamardProduct(delta, broadcastFunction(D_activation, layers[l].Activations));

                // Update weights and biases for the current layer (l)
                dw = matrixMultiply(delta, transposeMatrix(outputs[l - 1]));
                layers[l].W = addMatrix(layers[l].W, scaleMatrix(dw, LEARNING_RATE));
                layers[l].B = addMatrix(layers[l].B, scaleMatrix(delta, LEARNING_RATE));
            }

            // Backpropagation for the first layer (0)
            delta = matrixMultiply(transposeMatrix(layers[1].W), delta);
            delta = hadamardProduct(delta, broadcastFunction(D_activation, outputs[0]));

            // Update weights and biases for the first layer (0)
            dw = matrixMultiply(delta, transposeMatrix(current_input));
            layers[0].W = addMatrix(layers[0].W, scaleMatrix(dw, LEARNING_RATE));
            layers[0].B = addMatrix(layers[0].B, scaleMatrix(delta, LEARNING_RATE));
        }
        }

        //printf("\nEpoch %d - Cost: %f\n", ep, cost(layers, NUM_LAYERS, data,activation));
        // for (int l = 0; l < numLayers; l++) {
        //     printf("\n--Layer %d--\n", l);
        //     printf("Weights:\n");
        //     printMatrix(layers[l].W);
        //     printf("Biases:\n");
        //     printMatrix(layers[l].B);
        // }
    }

    return NULL; // You can return something meaningful, or void, based on your needs
}


void calculateRSquare(Layer *layers, dataset* data, float (*activation)(float))
{
    

    float sum_of_squares_total = 0.0;
    float sum_of_squares_residual = 0.0;
    float mean_y = 0.0;

    printf("\n----------------------------\n");
    for (size_t i = 0; i < SIZE_DATA; i++)
    {
        mean_y += data[i][PRED_COLUMN];
    }
    mean_y /= SIZE_DATA;


    for(size_t i=0; i< SIZE_DATA; i++)
    {
        float y_hat = getElementAt(forwardPropagate(layers,NUM_LAYERS, data[i],activation),0,0);\
        float y = data[i][1];

        printf("\n%f %f\n",y_hat, data[i][PRED_COLUMN]);
        sum_of_squares_total += (y - mean_y) * (y - mean_y);
        sum_of_squares_residual += (y - y_hat) * (y - y_hat);
       
    }

    float r2 = 1.0 - (sum_of_squares_residual / sum_of_squares_total);
    printf("\nR^2 SCORE : %f\n",r2);


}

#endif