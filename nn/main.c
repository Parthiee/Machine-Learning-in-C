#include <stdio.h>
#include "nn.h"



dataset or_train[] = {
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,1}

};

dataset *data = or_train;

int main()
{
    Layer l_1 = createLayer(INPUT,L1);
    Layer l_2 = createLayer(L1,L2);
    Layer layers[] = {l_1, l_2};

    forwardPropagate(layers,2, data[0]);
    printf("\nCost is %f \n",cost(layers,2,data));


    printf("\n---------main-------\n");
    printMatrix(l_1.W);
    printf("\n\n");
    printMatrix(l_2.W);
    printf("\n\n");
    
    printf("\n----------------\n");

    // backPropagate(layers,2,data);
    // Matrix* C = allocateMatrix(3,3);
    // randomizeMatrix(C);
    

    return 0;
}