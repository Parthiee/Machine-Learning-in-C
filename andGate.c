#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

// OR- Gate
double data[][3] ={
      {0,0,0},
      {0,1,0},
      {1,0,0},
      {1,1,1}
    };

#define traincount 10000
#define size sizeof(data)/sizeof(data[0])



double getRandom()
{
 
  //srand(time(NULL));

  return (double) rand() / RAND_MAX * 10.00f;
}

double sigmoid(double x)
{
    return (double) 1/(1+exp(-x));
}

double getMeanSquaredError(double w1, double w2, double bias)
{
  double cumError=0;
  

  for(int i=0; i<size; i++)
  {
  
    double y_hat = sigmoid(w1*data[i][0] + w2*data[i][1] + bias) ; 
    double y = data[i][2];
    cumError += (y - y_hat)*(y - y_hat);

  }
  cumError /= size;
 
  return cumError;

}

void optimizeWeight(double *w1, double *w2, double *bias, int learnCount, double delta, double learningRate)
{
  for(int i=0; i<learnCount; i++)
  {
    double dw1 = (getMeanSquaredError(*w1+delta, *w2, *bias) - getMeanSquaredError(*w1, *w2, *bias))/delta;
    double dbias = (getMeanSquaredError(*w1, *w2, *bias+delta) - getMeanSquaredError(*w1, *w2 ,*bias))/delta;
    double dw2 = (getMeanSquaredError(*w1, *w2+delta, *bias) - getMeanSquaredError(*w1, *w2, *bias))/delta;
    printf("\n%lf %lf %lf %lf\n",*w1,*w2,*bias,getMeanSquaredError(*w1,*w2,*bias));
    *w1 -= dw1*learningRate;
    *w2 -= dw2*learningRate;
    *bias -= dbias*learningRate;

  }
  return;
  
}


int main()
{
  // y = sigmoid(m*x + c);
  double w1 = getRandom();
  double w2 = getRandom();
  double bias = getRandom();
  printf("Initial weight w1: %lf, w2: %lf, Initial bias: %lf\n",w1,w2,bias);
  double delta = 0.001;
  double learningRate = 0.01;

  
  printf("Error : %lf\n",getMeanSquaredError(w1,w2,bias));
  optimizeWeight(&w1, &w2, &bias, 1000*1000,delta,learningRate);

  printf("\n%5s %10s","y'", "y");
  for(int i=0; i<size; i++)
  {

    double y_hat = sigmoid(w1*data[i][0] + w2*data[i][1] + bias) ;
    double y = data[i][2];
    printf("\n%lf  %lf \n", y_hat, y);
  }
  printf("\n");
  printf("Optimized w1:%lf, w2:%lf, Optimized bias: %lf\n",w1,w2,bias);
  printf("Error : %lf\n",getMeanSquaredError(w1,w2,bias));
  

  return 0;
}
