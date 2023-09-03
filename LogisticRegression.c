#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

double data[][2] ={
        {1, 0.3}, {2, 0.6}, {3, 0.8}, {4, 0.2}, {5, 0.9},
        {6, 0.4}, {7, 0.7}, {8, 0.1}, {9, 0.5}, {10, 0.75},
        {11, 0.35}, {12, 0.63}, {13, 0.82}, {14, 0.27}, {15, 0.91},
        {16, 0.42}, {17, 0.68}, {18, 0.15}, {19, 0.55}, {20, 0.78},
        {21, 0.38}, {22, 0.61}, {23, 0.88}, {24, 0.22}, {25, 0.93},
        {26, 0.47}, {27, 0.72}, {28, 0.12}, {29, 0.58}, {30, 0.76},
        {31, 0.33}, {32, 0.66}, {33, 0.85}, {34, 0.25}, {35, 0.94},
        {36, 0.45}, {37, 0.74}, {38, 0.11}, {39, 0.53}, {40, 0.77},
        {41, 0.31}, {42, 0.64}, {43, 0.87}, {44, 0.29}, {45, 0.92},
        {46, 0.49}, {47, 0.71}, {48, 0.14}, {49, 0.57}, {50, 0.79}
    };

int size = sizeof(data)/sizeof(data[0]);

double getRandom()
{
  time_t t;
  //srand(time(NULL));

  return (double) rand() / RAND_MAX * 10.00f;
}

double sigmoid(double x)
{
    return (double) 1/(1+exp(-x));
}

double getMeanSquaredError(double weight, double bias)
{
  double cumError=0;
  

  for(int i=0; i<size; i++)
  {
    //double y_hat = weight*data[i][0]  + bias;
    double y_hat = sigmoid(weight*data[i][0]  + bias);
    double y = data[i][1];
    cumError += (y - y_hat)*(y - y_hat);

  }
  cumError /= size;

  return cumError;

}

void optimizeWeight(double *weight, double *bias, int learnCount, double delta, double learningRate)
{
  for(int i=0; i<learnCount; i++)
  {
    double dweight = (getMeanSquaredError(*weight+delta, *bias) - getMeanSquaredError(*weight, *bias))/delta;
    *weight -= dweight*learningRate;
    double dbias = (getMeanSquaredError(*weight, *bias+delta) - getMeanSquaredError(*weight, *bias))/delta;

    *bias -= dbias*learningRate;

  }
  return;
  
}


int main()
{
  // y = m*x + c;
  double weight = getRandom()*10;
  double bias = getRandom()*10;
  printf("Initial weight: %lf, Initial bias: %lf\n",weight,bias);
  double delta = 0.001;
  double learningRate = 0.001;
  double lambda = 0.001;

  // for(int i=0; i<size; i++)
  // {
  //   double y_hat = weight*data[i][0];
  //   double y = data[i][1];
  //   printf("\n%lf  %lf \n", y_hat, y);
  // }
  
  printf("Error : %lf\n",getMeanSquaredError(weight,bias));
  optimizeWeight(&weight,&bias, 10000,delta,learningRate);

  printf("Optimized weight:%lf, Optimized bias: %lf\n",weight,bias);
  printf("Error : %lf\n",getMeanSquaredError(weight,bias));

    for(int i=0; i<size; i++)
  {
   double y_hat = sigmoid(weight*data[i][0] + bias);
   // double y_hat = weight*data[i][0]  + bias;
    double y = data[i][1];
    printf("\n%lf  %lf \n", y_hat, y);
  }
  
  

  return 0;
}
