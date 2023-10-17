#include <stdio.h>
#include "nn.h"



dataset or_train[] = {
    {0,0,1},
    {0,1,1},
    {1,0,1},
    {1,1,0}

};
dataset *data = or_train;

int main()
{
    Layer l_1 = createLayer(INPUT,L1);
    Layer l_2 = createLayer(L1,L2);
    Layer l_3 = createLayer(L2,L3);
    Layer layers[] = {l_1, l_2};

    
    printf("\nCost is %f \n",cost(layers,NUM_LAYERS,data));



    backPropagate(layers,NUM_LAYERS,data);


     printf("\nCost is %f \n",cost(layers,NUM_LAYERS,data));

     for(size_t i=0; i< SIZE_DATA; i++)
     {
        printf("\n%f %f\n",getElementAt(forwardPropagate(layers,NUM_LAYERS, data[i]),0,0), data[i][2]);
     }
     
    

    return 0;
}