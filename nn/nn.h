#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "matrix.h"

#define LAYER_SIZE 4 
#define INPUT_SIZE 5// NEURONS IN ONE LAYER

#define SIZE_DATA 4 // data
#define L1 4
#define L2 1
#define INPUT 2

typedef float dataset[3];



// typedef struct nn
// {
//    int input_size;
//    Matrix* weights;

// }Neuron;


// Neuron createNeuron()
// {
//     Neuron neu;
//     neu.input_size = INPUT_SIZE;
//     neu.weights = allocateMatrix(INPUT_SIZE, 1);
//     randomizeMatrix(neu.weights);

//     return neu;

// }

// Matrix generateWeightMatrix(Neuron* layer)
// {
//     Matrix* W = allocateMatrix(INPUT_SIZE, LAYER_SIZE);
//     for(size_t i=0; i<W->row; i++)
//     {
//             for(size_t j=0; j<W->col; j++)
//              {

//              } 
//     } 
// }


typedef struct 
{
    int numNodes;
    int numInputs;
    Matrix* W;
    Matrix* B;

}Layer;

Layer createLayer(int numInputs,int numNodes)
{
    Layer l;

    l.numInputs = numInputs;
    l.numNodes = numNodes;
    l.W = allocateMatrix(numInputs, numNodes);
    l.W =transposeMatrix(l.W);
    l.B = allocateMatrix(numNodes, 1);
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
        input = output;
       }

        // output = matrixMultiply(l_2.W,output);
        // output = addMatrix(output,l_2.B);
        // output = broadcastFunction(sigmoid,output);
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