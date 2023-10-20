#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

double data[][2] ={
{1.00  , 3.00},
{2.00 ,  5.00},
{3.00 ,  7.00},
{4.00 ,  9.00},
{5.00 , 11.00},
{6.00 , 13.00},
{7.00 , 15.00},
{8.00 , 17.00},
{9.00 , 19.00},
{10.00, 21.00},
};
 
#define traincount 10000
#define size sizeof(data)/sizeof(data[0])


double relu(double x) {
    if (x > 0) {
        return x;
    } else {
        return 0.0;
    }
}


double getRandomf()
{
  srand(time(0));

  return (double) rand() / RAND_MAX ;
}

double getMeanSquaredError(double weight, double bias)
{
  double cumError=0;
  

  for(int i=0; i<size; i++)
  {
    double y_hat = relu( weight*data[i][0]  + bias);
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
  double weight = getRandomf();
  double bias = getRandomf();
  printf("Initial weight: %lf, Initial bias: %lf\n",weight,bias);
  double delta = 0.0001;
  double learningRate = 0.001;

  
  printf("Error : %lf\n",getMeanSquaredError(weight,bias));
  optimizeWeight(&weight,&bias,10000,delta,learningRate);

  printf("Optimized weight:%lf, Optimized bias: %lf\n",weight,bias);
  printf("Error : %lf\n",getMeanSquaredError(weight,bias));

  for(int i=0; i<size; i++)
  {
    double y_hat = weight*data[i][0] + bias;
    double y = data[i][1];
    printf("\n%lf  %lf \n", y_hat, y);
  }
  
  

  return 0;
}
