#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

double data[][2] = {
    {0.1, 0.8},
    {0.2, 0.7},
    {0.3, 0.6},
    {0.4, 0.5},
    {0.5, 0.4},
    {0.6, 0.3},
    {0.7, 0.2},
    {0.8, 0.1},
    {0.9, 0.0},
    {1.0, 0.2},
    {1.1, 0.3},
    {1.2, 0.4},
    {1.3, 0.5},
    {1.4, 0.6},
    {1.5, 0.7},
    {1.6, 0.8},
    {1.7, 0.9},
    {1.8, 1.0},
    {-0.5, 0.9},
    {-0.4, 0.8},
    {-0.3, 0.7},
    {-0.2, 0.6},
    {-0.1, 0.5}
    
};

#define SIZE sizeof(data)/sizeof(data[0])
#define TRAINING_COUNT 1000*1000

double randomDouble()
{
    srand(time(0));
    return (double) rand()/RAND_MAX;
}

double sigmoid(double x)
{
    return (double) 1/(1+exp(-x));
}

double meanSquareError(double w, double bias)
{
    //Cost(y, y_pred) = -[y * log(y_pred) + (1 - y) * log(1 - y_pred)]

    double square_sum=0;

    for(int i=0; i<SIZE; i++)
    {
        double y_hat = sigmoid(w*data[i][0] + bias);
        double y = data[i][1];
        double error = -(y*log(y_hat) + (1-y)*log(1-y_hat));
        square_sum += error;
    }

    return square_sum/SIZE;
}

void gradientDecent(double *w1, double *bias, double delta, double learningRate)
{
  for(int i=0; i<TRAINING_COUNT; i++)
  {
    double dw1 = (meanSquareError(*w1+delta, *bias) - meanSquareError(*w1, *bias))/delta;
    double dbias = (meanSquareError(*w1, *bias+delta) - meanSquareError(*w1 ,*bias))/delta;

    printf("\n%lf %lf %lf\n",*w1,*bias,meanSquareError(*w1,*bias));
    *w1 -= dw1*learningRate;
    *bias -= dbias*learningRate;

  }
  return;
  
}

int main()
{
    double w = randomDouble();
    double bias = randomDouble();
    double delta = 1e-3;
    double learningRate = 1e-3;

    printf("\nInitial w: %lf bias: %lf\n", w, bias);
    gradientDecent(&w,&bias,delta,learningRate);
    for(int i=0; i<SIZE; i++)
    {
        double y_hat = sigmoid(w*data[i][0] + bias);
        double y = data[i][1];

        printf("\n%lf %lf\n",y_hat, y);

    }


    return 0;
}