/* 
Parallel Processing Final Project
Ahmed Darwich and Taylor Burgess
Dr. Tim O'Neil
November 28th 2022

Radix Sort:
*/


/* 
Predicate Function:
This function will test the first bit, and return an array of 1s and 0s 
known as the predicate. 1 will signify that the first bit is 0 while
0 will signify that the first bit is 1. It will calculate the number of ones
in the predicate and store it in d_NumOfOnes
Inputs:
    device_input: input array from which the predicate will be caluclated
    device_output: result of the predicate
*/
#include <stdio.h>
#include <helper_timer.h>

#define GRIDSIZE 1
#define BLOCK_SIZE 64

__global__ void calculatePredicate(int* device_input, int* device_output, int* d_NumOfOnes, int currentBit)
{
    // find where we are
    unsigned int currentThread = threadIdx.x;

    // if the first bit is 1, write 0 to output
    if((device_input[currentThread] & currentBit) == currentBit){
        device_output[currentThread] = 0;
    // else the first bit is 0, write 1 to output and add to d_NumOfOnes
    } else{
        device_output[currentThread] = 1;
        atomicAdd(d_NumOfOnes, 1);
    }
}

/*
Flip Bits Function
A function that flips all bit values given an input array and stores them in an output array.
*/
__global__ void flipBits(int *device_input,int *device_output)
{
    // find where we are
	int currentThread = threadIdx.x;
    // given that its all bits, we can just not the input value and store it in output
	device_output[currentThread] = !device_input[currentThread];
}

/*
Prefix Scan
A function that adjusts every element in the output list to the sum of 
all the previous elements including itself. 
*/
__global__ void prefixScan(int *device_input, int size)
{
    // find where we are
	int currentThread = threadIdx.x;
	int i;
    // iterate through and take the sume of the previous elements
	for(i=2; i <= size; i <<= 1)
	{
		
		if((currentThread + 1) % i == 0)
		{
			int offset = i >> 1;
			device_input[currentThread] += device_input[currentThread - offset];
		}
	}
	__syncthreads();

	// Down Sweep implementation
	device_input[size-1] = 0;
	int j;
    // iterate through array
	for(j = i >> 1; j >= 2; j >>= 1)
	{
		int offset = j >> 1;
        if((currentThread+1) % j == 0)
		{
			int currentElement = device_input[currentThread];
			device_input[currentThread] += device_input[currentThread - offset];
			device_input[currentThread - offset] = currentElement; 
		}

	}
}

/*
Radix Sort:
Determinees the new index for every element according to the following:
1.) For the ith element in the array:
    A.) If the predicate is 1, we move the element to the ith element in the predicateScan array. 
    B.) If the predicate is 0, we move the element to the index calculated by
            currentThread = value in ters_predict_scan + NumOfOnes
*/
__global__ void radixSort(int* device_input_array,int* device_output_array, int* device_predicate_array, int* device_predicate_scanned_array, int* device_predicate_NumOfOnes, int* device_flipped_predicate, int* device_flipped_predicate_scan)
{
    // find where we are
	int currentThread = threadIdx.x;
	if(device_predicate_array[currentThread] == 1)
	{
		int newThread = device_predicate_scanned_array[currentThread];
		device_output_array[newThread] = device_input_array[currentThread];
	}

	else
	{
		int newThread = device_flipped_predicate_scan[currentThread] + (*device_predicate_NumOfOnes);
		device_output_array[newThread] = device_input_array[currentThread];
	}
} 

int main(void)
{
    const int size = 64;
	// defining input array and fill it
	int *host_input_array = (int*) malloc(sizeof(int)*size);
	srand ( time(NULL) );
	for(int i = 0; i < size; i++){
        host_input_array[i] = rand() % 10;
    }

	// allocate memory on the host and device for the final sorted result array
	int* host_result_scan = (int*) malloc(sizeof(int) * size);
	int* device_result_scan;
	cudaMalloc(&device_result_scan, sizeof(int) * size);

	// allocate memory on the host and device for the flipped predicate results
	int* host_predicate_flipped_result = (int* )malloc(sizeof(int) * size);
	int* device_predicate_flipped_result;
	cudaMalloc(&device_predicate_flipped_result, sizeof(int) * size);

	//allocate memory on the host and device for the predicate result
	int* host_predicate_result = (int*) malloc(sizeof(int) * size);
	int* device_predicate_result;
	cudaMalloc(&device_predicate_result, sizeof(int) * size);

	//allocate memory on the device for the input array
	int* device_input_array;
	cudaMalloc(&device_input_array, sizeof(int) * size);
	cudaMemcpy(device_input_array, host_input_array, sizeof(int) * size, cudaMemcpyHostToDevice);

	//allocate memory on the device for the number of ones in the predicate result 
	int* device_NumOfOnes;
	int* host_NumOfOnes = (int*) malloc(sizeof(int));
	cudaMalloc(&device_NumOfOnes, sizeof(int));

	//allocate memory on the host for the prefixScan result array
	int* host_flipped_predicate_scan = (int*) malloc(sizeof(int) * size);
	int* device_result_predicate_scan;
	cudaMalloc(&device_result_predicate_scan, sizeof(int) * size);

	//allocate memory on host and device for output sorted array
	int* host_sort_result = (int*) malloc(sizeof(int) * size);
	int* device_sort_result;
	cudaMalloc(&device_sort_result, sizeof(int) * size);
	StopWatchLinux stw;

    // bitmap is a mask to be used in bitwise operations , initial value is 1 to test the first bit
    int bitmap = 1;
	stw.reset();
    stw.start();
    for(int k = 0; k < 4; k++) {
	    //set the numOfOnes to 0 at every iteration
	    cudaMemset(device_NumOfOnes, 0, sizeof(int));

	    // call the predicate kernel 
	    calculatePredicate<<<GRIDSIZE,BLOCK_SIZE>>>(device_input_array, device_predicate_result, device_NumOfOnes, bitmap);

	    //copy the predicate result and number of ones from the device to  the host
	    cudaMemcpy(host_predicate_result, device_predicate_result, sizeof(int) * size, cudaMemcpyDeviceToHost);
	    cudaMemcpy(host_NumOfOnes, device_NumOfOnes, sizeof(int), cudaMemcpyDeviceToHost);

	    //copy the predicate result from host to the device and store it in d_result scan. the change will be applied on the same array
	    cudaMemcpy(device_result_scan, host_predicate_result, sizeof(int) * size, cudaMemcpyHostToDevice);

	    //call the kernal function
	    prefixScan<<<GRIDSIZE,BLOCK_SIZE>>>(device_result_scan, size);

	    //copy the result back to the host 
	    cudaMemcpy(host_result_scan, device_result_scan, sizeof(int) * size, cudaMemcpyDeviceToHost);

	    //call the flip bits kernel on the device
	    flipBits<<<GRIDSIZE,BLOCK_SIZE>>>(device_predicate_result, device_predicate_flipped_result);

	    //copy the result to the host
	    cudaMemcpy(host_predicate_flipped_result, device_predicate_flipped_result, sizeof(int) * size,cudaMemcpyDeviceToHost);

	    //copy the !predicate from the host to the device and store it in device_result_predicate_scan
	    cudaMemcpy(device_result_predicate_scan, host_predicate_flipped_result, sizeof(int) * size, cudaMemcpyHostToDevice);

	    //call prefixScan upon device_result_flipped_scan, the change will be applied ont the same array
        // as this is mutating scan.
	    prefixScan<<<GRIDSIZE,BLOCK_SIZE>>>(device_result_predicate_scan, size);

	    //copy the result to the host host_flipped_predicate_scan
	    cudaMemcpy(host_flipped_predicate_scan, device_result_predicate_scan, sizeof(int) * size, cudaMemcpyDeviceToHost);

	    //call the radix sort kernel
	    radixSort<<<GRIDSIZE,BLOCK_SIZE>>>(device_input_array, device_sort_result, device_predicate_result, device_result_scan, device_NumOfOnes, device_predicate_flipped_result, device_result_predicate_scan);
	
        //copy the sorted list back to the host and print it 
	    cudaMemcpy(host_sort_result, device_sort_result, sizeof(int) * size, cudaMemcpyDeviceToHost);

	    //update the mask to test the next bit 
	    bitmap <<= 1;

	    // update the input array for every iteration
	    memcpy(host_input_array, host_sort_result, sizeof(int) * size);
	    // update the input array on the device
	    cudaMemcpy(device_input_array, host_input_array, sizeof(int) * size, cudaMemcpyHostToDevice);
    }
	stw.stop();

    for(int i = 0; i < size; i++){
        printf("Value: %d\n", host_sort_result[i]);
    }
	printf("Radix Sort Analyis: \nSize of Array: %d, Time to sort: %f ms.\n", size, stw.getTime());

  	cudaFree(device_input_array);
  	cudaFree(device_sort_result);
  	cudaFree(device_result_scan);
  	cudaFree(device_predicate_result);
  	cudaFree(device_NumOfOnes);
  	cudaFree(device_predicate_flipped_result);
  	cudaFree(device_result_predicate_scan);
	return 0;

	
}