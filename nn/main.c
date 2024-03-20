#include <stdio.h>
#include "nn.h"
#include "matrix.h"
#include "csv.h"
#include <string.h>
#include <stdlib.h>
#include "glob.h"

char *filename;
int numInputs;
int numHidden;
int *hiddenLayer;
char *activation;

float (*function)(float x);
float (*function_derivative)(float x);

int main(int argc, char** argv)
{



   if (argc < 4)
   {
        printf(" \n./main <filename> <input cols> <linear/sigmoid/relu> <no.of hidden layers> <no of nodes in hidden layers> \n");
        exit(1);

   }

   else
    {
        filename =(char*) malloc(FILENAME_MAX);
        activation =(char*) malloc(FILENAME_MAX);

        strcpy(filename,argv[1]);
        numInputs = atoi(argv[2]);
        numHidden = atoi(argv[4]);
        strcpy(activation,argv[3]);

        hiddenLayer = (int*) malloc(sizeof(int)*numHidden);
        for(int i=5,j=0; (i< argc) && (j <argc); i++,j++)
        {
            hiddenLayer[j] = atoi(argv[i]);

        }

        printf("\nFilename : %s ",filename);
        printf("\nNumInputs : %d ",numInputs);
        printf("\nNumHidden : %d ",numHidden);
        for(int j=0; j <numHidden; j++)
        {
            printf("\nHidden Layer %d : %d",j,hiddenLayer[j]);

        }

        if(!strcmp(activation,"relu"))
        {
            function = relu;
            function_derivative = relu_derivative;
        }

        else if(!strcmp(activation,"sigmoid"))
        {
            function = sigmoid;
            function_derivative = sigmoid_derivative;

        }

        else if(!strcmp(activation,"linear"))
        {
            function = linear;
            function_derivative = linear_derivative;

        }

  
    
    }

    
    DataFrame *t = createDataFrame(filename);
    dataset *data = t->df;
    Layer *layers = (Layer*) malloc(sizeof(Layer)*(numHidden+2));

    // Layer l_1 = createLayer(numInputs,L1);
    // Layer l_2 = createLayer(L1,L2);
    // Layer l_3 = createLayer(L2,OUTPUT);

    
    for(int i=0; i<numHidden+2;i++)
    {
        printf("\nWorks....");

        if(i==0)
        {
    
            layers[i] = createLayer(numInputs,hiddenLayer[0]);
            continue;
        }

        else if(i==numHidden)
        {
            layers[i] = createLayer(hiddenLayer[i-1],1);
            continue;
        }

        else if(i==numHidden+1)
        {

            layers[i] = createLayer(1,1);
            continue;
        }

        layers[i] = createLayer(hiddenLayer[i-1],hiddenLayer[i]);
 
    }

     for (int l = 0; l < (numHidden +2); l++) {
            printf("\n--Layer %d--\n", l);
            printf("Weights:\n");
            printMatrix(layers[l].W);
            printf("Biases:\n");
            printMatrix(layers[l].B);
        }

    printf("\nCost is %f \n",cost(layers,numHidden+2,data,function,numInputs));
    backPropagate(layers,numHidden+2,data,function, function_derivative, numInputs, numHidden);
    calculateRSquare(layers,data, function,numInputs,numHidden+2);
    printf("\nFinal Cost is %f \n",cost(layers,numHidden+2,data,function,numInputs));
    
    freeMemory(layers,numHidden+2);
    return 0;
}