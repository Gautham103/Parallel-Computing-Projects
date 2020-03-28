// -----------------------------------------------------------------
// Header file with routines to:
// - initialize the list that needs to be sorted
// - check that the list is sorted
//
#include "mpi.h"
#include <stdlib.h>

#ifndef VERBOSE
#define VERBOSE 0                       // Use VERBOSE to control output 
#endif

// Allocate and initialize local list
// Input: 
//   list_size 	- size of list
//   type 	- initialization type (elements in increasing  order, 
//  		  decreasing order, random, other types can be added)
//   my_id	- process rank 
//   num_procs	- number of MPI processes
// Output: 
//   list	- integer array of size list_size containing elements of
//   		  list
//
int * initialize_list(int list_size, int type, int my_id, int num_procs) {
    int j;
    int * list = (int *) calloc(list_size, sizeof(int));
    switch (type) {
	case -1:	// Elements are in descending order
	    for (j = 0; j < list_size; j++) {
		list[j] = (num_procs-my_id)*list_size-j;
	    }
	    break;
	case -2:	// Elements are in ascending order
	    for (j = 0; j < list_size; j++) {
		list[j] = my_id*list_size+j+1;
	    }
	    break;
	default: 
	    srand48(type + my_id); 
	    list[0] = lrand48() % 100;
	    for (j = 1; j < list_size; j++) {
		list[j] = lrand48() % 100;
	    }
	    break;
    }
    return list;
}

// Check if list is sorted. 
// Each process verifies that its local list is sorted in ascending order.
// The process also checks that its list has values larger than or equal to
// the largest value on the process before it (i.e., on process (my_id-1).
// Prints result of error check if VERBOSE > 1 
// Input: 
//   list_size 	- size of list
//   type 	- initialization type (elements in increasing  order, 
//  		  decreasing order, random, other types can be added)
//   my_id	- process rank 
//   num_procs	- number of MPI processes
//
void check_list(int *list, int list_size, int my_id, int num_procs) {
    int tag = 0;
    int max_nbr = -1;		// Assumes list contains non-negative integers; 6-21-2017
    int error, local_error;
    int j, my_max;
    MPI_Status status; 
    // Receive largest list value from process with rank (my_id-1)
    if (my_id-1 >= 0) {
	MPI_Recv(&max_nbr, 1, MPI_INT, my_id-1, tag, MPI_COMM_WORLD, &status);
	// Good practice to check status!
    }
    // Check that the local list is sorted and that elements are larger than 
    // or equal to the largest on process with rank (my_id-1)
    // (error is set to 1 if a pair of elements is not sorted correctly)
    local_error = 0;
    if (list_size > 0) {
	if (list[0] < max_nbr) local_error = 1; 
	for (j = 1; j < list_size; j++) {
	    if (list[j] < list[j-1]) local_error = 1;
	}
	my_max = list[list_size-1];
    } else {					// Modified 6-21-2017
        my_max = max_nbr;			// Modified 6-21-2017
    }
    if (VERBOSE > 1) {
	printf("[Proc: %0d] check_list: local_error = %d\n", my_id, local_error);
    }
    // Send largest list value to process with rank (my_id+1)
    if (my_id+1 < num_procs) {
	MPI_Send(&my_max, 1, MPI_INT, my_id+1, tag, MPI_COMM_WORLD);
	// Good practice to check status!
    }
    // Collect errors from all processes
    // error = 0;
    MPI_Reduce(&local_error, &error, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (my_id == 0) {
	if (error == 0) {
	    printf("[Proc: %0d] Congratulations. The list has been sorted correctly.\n", my_id);
	} else {
	    printf("[Proc: %0d] Error encountered. The list has not been sorted correctly.\n", my_id);
	}
    }
}
