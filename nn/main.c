#include <stdio.h>
#include "nn.h"
#include "matrix.h"
#include "csv.h"



// dataset or_train[] = {
//     {0,0,1},
//     {0,1,0},
//     {1,0,0},
//     {1,1,0}

// };



// int main2()
// {
  

//    DataFrame *t = createDataFrame("score_updated.csv");
//    dataset *data = t->df;


// }

int main()
{
    DataFrame *t = createDataFrame("score_updated.csv");
    dataset *data = t->df;

    Layer l_1 = createLayer(INPUT,L1);
    Layer l_2 = createLayer(L1,L2);
    Layer l_3 = createLayer(L2,L3);
    // Layer l_4 = createLayer(L3,L4);
    // Layer l_5 = createLayer(L4,L5);
    Layer layers[] = {l_1, l_2, l_3};

    
    printf("\nCost is %f \n",cost(layers,NUM_LAYERS,data,linear));
    backPropagate(layers,NUM_LAYERS,data,linear, linear_derivative);
    printf("\nCost is %f \n",cost(layers,NUM_LAYERS,data,linear));
    calculateRSquare(layers,data, linear);
    

    return 0;
}