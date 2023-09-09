#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#define size 4
#define TRAINING_COUNT 100

typedef double dataset[3];
typedef struct 
{
    double or_w1, or_w2, or_bias;
    double nand_w1, nand_w2, nand_bias;
    double and_w1, and_w2, and_bias;

}xor;

xor model;

typedef struct {
    double d_or_w1, d_or_w2, d_or_bias;
    double d_nand_w1, d_nand_w2, d_nand_bias;
    double d_and_w1, d_and_w2, d_and_bias;

} differentials;


dataset or_train[] = {
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,1}

};

dataset and_train[] = {
    {0,0,0},
    {0,1,0},
    {1,0,0},
    {1,1,1}

};

dataset nand_train[] = {
    {0,0,1},
    {0,1,1},
    {1,0,1},
    {1,1,0}

};

dataset *data = nand_train;

double sigmoid(double x)
{

    return (double) 1/(1+exp(-x));
}

double forward(xor model, double x1, double x2)
{
    double a = sigmoid(x1*model.or_w1 + x2*model.or_w2 + model.or_bias);
    double b = sigmoid(x1*model.nand_w1 + x2*model.nand_w2 + model.nand_bias);
    return sigmoid(model.and_w1*a + model.and_w2*b + model.and_bias);

}


double getRandom()
{
 
  srand(time(NULL));
  return (double) rand() / RAND_MAX * 10.00f;
}

double getMeanSquaredError(xor model)
{
  double cumError=0;
  

  for(int i=0; i<size; i++)
  {
  
    double y_hat = forward(model,(data)[i][0], (data)[i][1]) ; 
    double y = (data)[i][2];
    cumError += (y - y_hat)*(y - y_hat);

  }
  cumError /= size;

  return cumError;

}

void printParameters(xor *model)
{
    printf("\n<%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf> || %lf\n",
    model->or_w1, model->or_w2, model->or_bias,
    model->nand_w1, model->nand_w2, model->nand_bias,
    model->and_w1, model->and_w2, model->and_bias,
    getMeanSquaredError(*model));

}

void calculateDifference(xor *model, differentials *diff, double learningRate)
{
    model->or_w1 -= learningRate*diff->d_or_w1;
    model->or_w2 -= learningRate*diff->d_or_w2;
    model->or_bias -= learningRate*diff->d_or_bias;
    model->nand_w1 -= learningRate*diff->d_nand_w1;
    model->nand_w2 -= learningRate*diff->d_nand_w2;
    model->nand_bias -= learningRate*diff->d_nand_bias;
    model->and_w1 -= learningRate*diff->d_and_w1;
    model->and_w2 -= learningRate*diff->d_and_w2;
    model->and_bias -= learningRate*diff->d_and_bias;

}
xor gradientDescent(xor *model, double delta, double learningRate)
{
  differentials diff;
  double error_before, error_after, temp;

  for(int i=0; i<TRAINING_COUNT; i++)
  {
//     error_before = getMeanSquaredError(*model);
//     temp = model->or_w1;
//     model->or_w1 += delta;
//     error_after = getMeanSquaredError(*model);
//     model->or_w1 = temp;
//     diff.d_or_w1 = (error_before - error_after)/delta;
//     printf("\n<<<%lf , %lf>>>\n",error_after,error_after);

//     temp = model->or_w2;
//     model->or_w2 += delta;
//     error_after = getMeanSquaredError(*model);
//     model->or_w2 = temp;
//     diff.d_or_w2 = (error_before - error_after)/delta;
//     printf("\n<<<%lf , %lf>>>\n",error_after,error_after);

//     temp = model->or_bias;
//     model->or_bias += delta;
//     error_after = getMeanSquaredError(*model);
//     model->or_bias = temp;
//     diff.d_or_bias = (error_before - error_after)/delta;
//     printf("\n<<<%lf , %lf>>>\n",error_after,error_after);


//     temp = model->nand_w1;
//     model->nand_w1 += delta;
//     error_after = getMeanSquaredError(*model);
//     model->nand_w1 = temp;
//     diff.d_nand_w1 = (error_before - error_after)/delta;
//     printf("\n<<<%lf , %lf>>>\n",error_after,error_after);


//     temp = model->nand_w2;
//     model->nand_w2 += delta;
//     error_after = getMeanSquaredError(*model);
//     model->nand_w2 = temp;
//     diff.d_nand_w2 = (error_before - error_after)/delta;
//     printf("\n<<<%lf , %lf>>>\n",error_after,error_after);


//     temp = model->nand_bias;
//     model->nand_bias += delta;
//     error_after = getMeanSquaredError(*model);
//     model->nand_bias = temp;
//     diff.d_nand_bias = (error_before - error_after)/delta;
//     printf("\n<<<%lf , %lf>>>\n",error_after,error_after);

//     temp = model->and_w1;
//     model->and_w1 += delta;
//     error_after = getMeanSquaredError(*model);
//     model->and_w1 = temp;
//     diff.d_and_w1 = (error_before - error_after)/delta;
//     printf("\n<<<%lf , %lf>>>\n",error_after,error_after);

//     temp = model->and_w2;
//     model->and_w2 += delta;
//     error_after = getMeanSquaredError(*model);
//     model->and_w2 = temp;
//     diff.d_and_w2 = (error_before - error_after)/delta;
// printf("\n<<<%lf , %lf>>>\n",error_after,error_after);

//     temp = model->and_bias;
//     model->and_bias += delta;
//     error_after = getMeanSquaredError(*model);
//     model->and_bias = temp;
//     diff.d_and_bias = (error_before - error_after)/delta;
//     printf("\n<<<%lf , %lf>>>\n",error_after,error_after);

    calculateDifference(model, &diff, learningRate);
    printParameters(model);

    

  }
  return *model;
}

int main()
{
  double delta = 0.001;
  double learningRate = 0.001;

  model.and_bias=getRandom();
  model.and_w1=getRandom();
  model.and_w2=getRandom();

  model.nand_bias=getRandom();
  model.nand_w1=getRandom();
  model.nand_w2=getRandom();

  model.or_bias=getRandom();
  model.or_w1=getRandom();
  model.or_w2=getRandom();
  printParameters(&model);
  printf("sfedff");


  xor t = gradientDescent(&model,delta, learningRate);

  printf("------------------------------------------");
  printParameters(&t);

 
  



    return 0;
}