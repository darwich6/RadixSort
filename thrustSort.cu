/* 
Parallel Processing Final Project
Ahmed Darwich and Taylor Burgess
Dr. Tim O'Neil
November 28th 2022

Thurst Sort for Comparison.
*/

#include <thrust/sort.h>
#include <thrust/functional.h>

#include <iostream>
#include <stdlib.h>
#include <helper_timer.h>

int main(void)
{
    const int N = 64;
    int A[N];
    StopWatchLinux stw;

    srand ( time(NULL) );
    for(int i = 0; i < N; i++){
        A[i] = rand() % 10;
    }

    stw.reset();
    stw.start();
    thrust::stable_sort(A, A + N, thrust::less<int>());
    stw.stop();

    for(int i = 0; i < N; i++){
        printf("Element: %d, Value: %d \n", i, A[i]);
    }
    printf("Performing Thrust Sort computation: %f ms for an array size of: %d\n", stw.getTime(), N);
}