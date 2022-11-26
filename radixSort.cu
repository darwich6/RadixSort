/* 
Parallel Processing Final Project
Ahmed Darwich and Taylor Burgess
Dr. Tim O'Neil
November 28th 2022

Radix Sort:
*/

/*
   For each bit position in a value, partition the elements so that all elements
   with a 0 in that bit position precede those with a 1 in that position, using a 
   stable sort. A sort is stable if the sort preserves the relative order of equal
   elements. When all the bits have been processed, the array would be sorted. Also
   Note that since this is a device function, after each patitioning step, the threads
   must be synchronized so that the array is ready for the next step.
*/
__device__ void radixSort(unsigned int *values){
    
    // For each bit in a value, call the partitionByBit function 
    for(int bit = 0; bit < 32; ++bit){
        paritionByBit(values, bit);
        __syncthreads();
    }
}

/*
   PrefixSum(x[]), where x[] is an array of integers, replaces all of x by the 
   prefix sums of the elements of x. The prefix sum of an element in an array is
   the sum of all elements up to and inlcuding that element. It is inclusive. 

   This function will return the value in the place of the current value. Once 
   it is completed, all threads should replace the elements of x[] with its prefixSums
*/
template<class T>
__device__ void prefixSum(T *x){
    
    // id of current thread
    unsigned int currentThread = threadIdx.x;
    // total number of threads in this block
    unsigned int numOfThreads = blockDim.x;
    // distance between elements that are beigng added
    unsigned int offset;
    
    // for each element in the array given the offset
    for(offset = 1; offset < numOfThreads; offset *= 2){
        T t;
        
        // if we are in an actual array position, store the value in T
        if(currentThread >= offset){
            t = x[i-offset];
        }

        __syncthreads();
        
        // add the sums of previous elements
        if(currentThread >= offset){
            x[currentThread] = x[currentThread] + x[currentThread - 1];
        }
        __syncthreads();
    }
    // return the final element of the array
    return x[currentThread];
}


/*
   partitionByBit

   This function is executed by every thread. It takes an array of integer 
   values and a bit position. It will partition the array such that for all values
   [i], i = 0,...,n-1, the value of bits b in each element values[k] for k < i is
   <= the value of bit b in values[i] and if bit b in value[j] == bit b in values[i]
   and j < i, then after the partition, the two elements will be in the same 
   relative order
*/
__device__ void partitionByBit(unsigned int *values, unsigned int bit){

    // id of currentThread
    unsigned int currentThread = threadIdx.x;
    // total number of threads in block
    unsigned int numOfThreads = blockDim.x;
    // value of the current element
    unsigned int currentElement = values[currentThread];
    // value of bit at the inputted position
    unsigned int currentBitValue = (currentElement >> bit) & 1;

    /* Replace the values of the inputted array so that the values
       at each index is the bit value in the current element */
    values[currentThread] = currentBitValue;
    __syncthreads();

    // Find the number of true bits up to and including values[i]
    unsigned int trueBitsBefore = prefixSum(values);
    // since we have a synthreads call at the last iteration in prefixSum. 
    // We know that the array has its prefixSum computed and the last element
    // is the sum of all elements
    unsigned int trueBitsTotal = values[numOfThreads - 1];
    // Find the number of 0-bits
    unsigned int falseBitsTotal = numOfThreads - trueBitsTotal;
    __syncthreads();

    /* 
    The currentElement now needs to be put back in the correct position
    The array has to satisfy the condition that all values with a 0 in 
    the current bit position must precede all those with a 1 in that position
    and it must be a stable sort.

    So, if the current element had a 1 in the current bit position before,
    it must be in that position such at all other elements  that had a 0
    precede it, and all other elemenets that had a 1 in that bit and for which
    they are different eleemnts, must precede it. So, if the current element
    had a 1, it must go into the index trueBitsBefore-1 + falseBitsTotal, which
    is the sum of the 0-bits and 1-bits that preceded it before.

    If the current element has a 0 in the current bit position, then it has to 
    move down the array before all next eleements that have a 1 in the current bit.
    Since there are trueBitsBefore before the current eleemnt, it would have to 
    move to current position of the current element - trueBitsBefore 
    */
    if(currentBitValue){
        values[trueBitsBefore - 1 + falseBitsTotal] = currentElement;
    else{
        values[currentThread trueBitsBefore] = currentElement;
    }


