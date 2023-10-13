#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "matrix.h"


#define SIZE_DATA 4 // data
#define L1 3
#define L2 1
#define L3 1
#define INPUT 2
#define EPOCH 1000000
#define LEARNING_RATE 0.001

typedef float dataset[3];


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


Matrix* forwardPropagate(Layer* layers, int numLayers, dataset ip)
{
    Matrix* input = allocateMatrix(INPUT,1);
    Matrix* output;

   
       setElementAt(input,0,0, ip[0]);
       setElementAt(input,1,0, ip[1]); // Manually change this

       // Y = W*x
       for(size_t j=0; j<numLayers; j++)
       {
        output = matrixMultiply(layers[j].W,input);
        output = addMatrix(output,layers[j].B);
        output = broadcastFunction(sigmoid,output);
        copyMatrix(layers[j].Activations,output);
        input = output;
       }
       return output;
    

    
}

float cost(Layer* layers, int numLayers, dataset* data)
{
    float cumError=0;
    for(size_t i=0; i<SIZE_DATA; i++)
    {
        Matrix* output = forwardPropagate(layers,numLayers,data[i]);
        float y_hat = getElementAt(output,0,0);
        float y = data[i][2];
        cumError += (y - y_hat)*(y - y_hat);
    }

  cumError /= SIZE_DATA;

  return cumError;

}



// Matrix* backPropagate(Layer* layers, int numLayers, dataset* data)
// {

//     Matrix *z_values, *activation_function_derivative;
//     for(size_t ep=0; ep< EPOCH; ep++)
//     {
//             for(size_t l=numLayers-1; l>0; l--)
//             {
//                if(l<0) break;
//                float do_cost;
//                int input = numLayers-2;
//                Layer input_layer = layers[input];

//                 // activation matrix
//             printf("\ncurrent weight : %ld x %ld\n",layers[l].W->row,layers[l].W->col);
//             printf("\nprev activation : %ld x %ld\n",input_layer.Activations->row,input_layer.Activations->col);
//                z_values = matrixMultiply(layers[l].W, input_layer.Activations);
//                z_values = addMatrix(z_values,layers[l].B);
//                activation_function_derivative = broadcastFunction(D_sigmoid,z_values);

            
//                printf("\n-------- Z ---------\n");
//                printMatrix(z_values);

//             }
//     }


// }


Matrix* backPropagate(Layer* layers, int numLayers, dataset* data) {
    for (size_t ep = 0; ep < EPOCH; ep++) {

        for (size_t i = 0; i < SIZE_DATA; i++) {
            // Forward propagation
            Matrix* outputs[numLayers];
            Matrix* current_input = allocateMatrix(INPUT, 1);
            setElementAt(current_input, 0, 0, data[i][0]);
            setElementAt(current_input, 1, 0, data[i][1]);

            outputs[0] = matrixMultiply(layers[0].W, current_input);
            outputs[0] = addMatrix(outputs[0], layers[0].B);
            outputs[0] = broadcastFunction(sigmoid, outputs[0]);
            copyMatrix(layers[0].Activations, outputs[0]);

            for (size_t l = 1; l < numLayers; l++) {
                outputs[l] = matrixMultiply(layers[l].W, outputs[l - 1]);
                outputs[l] = addMatrix(outputs[l], layers[l].B);
                outputs[l] = broadcastFunction(sigmoid, outputs[l]);
                copyMatrix(layers[l].Activations, outputs[l]);
            }

            // Calculate the error at the output layer
            float y_hat = getElementAt(outputs[numLayers - 1], 0, 0);
            float y = data[i][2];
            float output_error = y_hat - y;

            // Backpropagation
            Matrix* delta = allocateMatrix(1, 1);
            setElementAt(delta, 0, 0, output_error);

            for (size_t l = numLayers - 1; l > 0; l--) {
                // Calculate delta weights and delta biases
                if(l == 0)
                { 
                
                Matrix* delta_weights = matrixMultiply(delta, transposeMatrix(current_input));
                Matrix* delta_bias = delta;

                // Update weights and biases for the current layer
                layers[l].W = subtractMatrix(layers[l].W, scaleMatrix(delta_weights, LEARNING_RATE));
                layers[l].B = subtractMatrix(layers[l].B, scaleMatrix(delta_bias, LEARNING_RATE));

                // Calculate the delta for the next layer
                delta = matrixMultiply(transposeMatrix(layers[l].W), delta);
                delta = hadamardProduct(delta, broadcastFunction(D_sigmoid, current_input));
            
                }
                else{
                Matrix* delta_weights = matrixMultiply(delta, transposeMatrix(outputs[l - 1]));
                Matrix* delta_bias = delta;

                // Update weights and biases for the current layer
                layers[l].W = subtractMatrix(layers[l].W, scaleMatrix(delta_weights, LEARNING_RATE));
                layers[l].B = subtractMatrix(layers[l].B, scaleMatrix(delta_bias, LEARNING_RATE));

                // Calculate the delta for the next layer
                delta = matrixMultiply(transposeMatrix(layers[l].W), delta);
                delta = hadamardProduct(delta, broadcastFunction(D_sigmoid, outputs[l - 1]));
                }
            }
        }

        printf("\n--Layer 0--\n");
        printMatrix(layers[0].W);
        printf("\n");
        printMatrix(layers[0].B);


        printf("\n--Layer 1--\n");
        printMatrix(layers[1].W);
        printf("\n");
        printMatrix(layers[1].B);
        

        // printf("\n--Layer 2--");
        // printMatrix(layers[2].W);
        // printf("\n");
        // printMatrix(layers[2].B);
        // printf("\n");
    }

    return NULL; // You can return something meaningful, or void, based on your needs
}

