#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include "glob.h"


#define MAX_LINE 1024

typedef struct DataFrame
{
    float df[SIZE_DATA][NUM_COLS];

}DataFrame;

void printDataFrame(const DataFrame *data)
{
    for(int row=0; row<SIZE_DATA; row++)
    {
        printf("\n%d: ",row);
        for(int col=0; col<NUM_COLS; col++)
        {
            printf("%f ",data->df[row][col]);
        }

        
    }
}

DataFrame* createDataFrame(char* filepath)
{
    FILE* file = fopen(filepath, "r");
    DataFrame* data = malloc(sizeof(DataFrame));
    char line[MAX_LINE];
    int row=0, col=0;
    char *token;

    while(fgets(line,MAX_LINE,file) && row < SIZE_DATA)
    {
        
        token = strtok(line, ",");

        col=0;
        for(int index = 0; index< NUM_COLS; index++)
        {
            printf("%s ",token);
            data->df[row][col] = (float) atof(token);
            token = strtok(NULL, ",");
            col++;
        }

        row++;
    }

    printDataFrame(data);
    fclose(file);
    return data;
}



