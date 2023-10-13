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
#define EPOCH 1000


typedef float dataset[3];


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


Matrix* backPropagate(Layer* layers, int numLayers, dataset* data)
{

    for(size_t ep=0; ep< EPOCH; ep++)
    {
        for(size_t i=0; i< SIZE_DATA; i++)
        {
            for(size_t l=numLayers-1; l>=0; l--)
            {
               // layers[l]
            }


        }
    }


}