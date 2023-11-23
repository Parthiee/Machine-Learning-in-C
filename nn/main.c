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

int main(int argc, char** argv)
{


    /*
        ./main <filename> <input cols> <no.of hidden layers> <no of nodes in hidden layers>
                            INPUT       
    */

   if (argc < 4)
   {
        printf(" \n./main <filename> <input cols> <no.of hidden layers> <no of nodes in hidden layers> \n");
        exit(1);

   }

   else
    {
        filename =(char*) malloc(FILENAME_MAX);
        strcpy(filename,argv[1]);
        numInputs = atoi(argv[2]);
        numHidden = atoi(argv[3]);
    

        hiddenLayer = (int*) malloc(sizeof(int)*numHidden);
        for(size_t i=4,j=0; (i< argc) && (j <argc); i++,j++)
        {
            hiddenLayer[j] = atoi(argv[i]);

        }

        printf("\nFilename : %s ",filename);
        printf("\numInputs : %d ",numInputs);
        printf("\numHidden : %d ",numHidden);
        for(size_t j=0; j <numHidden; j++)
        {
            printf("\nHidden Layer %lu : %d",j,hiddenLayer[j]);

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

    
    printf("\nCost is %f \n",cost(layers,numHidden+2,data,linear,numInputs));
    backPropagate(layers,numHidden+2,data,linear, linear_derivative, numInputs);
    calculateRSquare(layers,data, linear,numInputs,numHidden+2);
    printf("\nFinal Cost is %f \n",cost(layers,numHidden+2,data,linear,numInputs));
    
    freeMemory(layers,numHidden+2);
    return 0;
}