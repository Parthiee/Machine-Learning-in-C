#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "matrix.h"

/*
    AND, NAND works for very large num of neurons

*/


#define SIZE_DATA 4 // data
#define L1 70// 100 neurons, works
#define L2 1
#define L3 1
#define INPUT 2
#define EPOCH 1000000
#define LEARNING_RATE 0.01
#define NUM_LAYERS 2

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
        if(cost(layers, numLayers, data) < 0.00001)
         {
            break;
         }
        for (size_t i = 0; i < SIZE_DATA; i++) {
            // Forward propagation
           // printf("\n%ld vaati ulla poiruchu\n", i);
            Matrix* outputs[numLayers];
            Matrix* current_input = allocateMatrix(INPUT, 1);
            setElementAt(current_input, 0, 0, data[i][0]);
            setElementAt(current_input, 1, 0, data[i][1]);

            //printf("\n%ld computing activations\n", i);
            outputs[0] = matrixMultiply(layers[0].W, current_input);
            
            outputs[0] = addMatrix(outputs[0], layers[0].B);
  
            outputs[0] = broadcastFunction(sigmoid, outputs[0]);\

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
            float output_error = y - y_hat;

            // Backpropagation for the last layer (numLayers - 1)
            Matrix* delta = allocateMatrix(1, 1);
            setElementAt(delta, 0, 0, output_error);

            // Update weights and biases for the output layer (numLayers - 1)
            Matrix* dw = matrixMultiply(delta, transposeMatrix(outputs[numLayers - 2]));
            
            layers[numLayers - 1].W = addMatrix(layers[numLayers - 1].W, scaleMatrix(dw, LEARNING_RATE));
            layers[numLayers - 1].B = addMatrix(layers[numLayers - 1].B, scaleMatrix(delta, LEARNING_RATE));
            
            // Backpropagation for the first layer (0)
            delta = matrixMultiply(transposeMatrix(layers[numLayers - 1].W), delta);
            delta = hadamardProduct(delta, broadcastFunction(D_sigmoid, outputs[numLayers - 2]));

            // Update weights and biases for the first layer (0)
            dw = matrixMultiply(delta, transposeMatrix(current_input));

            Matrix* scales = scaleMatrix(dw, LEARNING_RATE);
            layers[0].W = addMatrix(layers[0].W, scales );
            scales = scaleMatrix(delta, LEARNING_RATE);
            layers[0].B = addMatrix(layers[0].B, scales);
           
        }   
        printf("\nCost : %f", cost(layers, numLayers, data));
        // printf("\n--Layer 0--\n");
        // printMatrix(layers[0].W);
        // printf("\n");
        // printMatrix(layers[0].B);

        // printf("\n--Layer 1--\n");
        // printMatrix(layers[1].W);
        // printf("\n");
        // printMatrix(layers[1].B);
    }

    return NULL; // You can return something meaningful, or void, based on your needs
}




