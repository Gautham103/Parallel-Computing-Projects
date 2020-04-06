// -----------------------------------------------------------------------------
// Hypercube Quicksort to sort a list of integers distributed across processors
// MPI-based implementation
//
// Routines:
//   main	- main program that implements hypercube quicksort algorithm
//
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "qsort_hypercube.h"

#define MAX_LIST_SIZE_PER_PROC	268435456

#ifndef VERBOSE
#define VERBOSE 0			// Use VERBOSE to control output
#endif

// Routines --------------------------------------------------------------------
//
// Computes the rank of neighbor process along dimension k (k > 0) of
// the hypercube. Rank is computed by flipping the kth bit of rank/id of
// this process
//
int neighbor_along_dim_k(int my_id, int k) {
    int mask = 1 << (k-1);
    return (my_id ^ mask);
}

// Merge two sorted lists and return the merged list
// Input:
//   list1, list1_size	- first list and its size
//   list2, list2_size	- second list and its size
// Output:
//   list		- merged list (size not returned!)
//
int * merged_list(int * list1, int list1_size, int * list2, int list2_size) {
    int * list = (int *) calloc((list1_size+list2_size), sizeof(int));
    int idx1 = 0;
    int idx2 = 0;
    int idx = 0;
    while ((idx1 < list1_size) && (idx2 < list2_size)) {
	if (list1[idx1] <= list2[idx2]) {
	    list[idx] = list1[idx1];
	    idx++; idx1++;
	} else {
	    list[idx] = list2[idx2];
	    idx++; idx2++;
	}
    }
    while (idx1 < list1_size) {
	list[idx] = list1[idx1];
	idx++; idx1++;
    }
    while (idx2 < list2_size) {
	list[idx] = list2[idx2];
	idx++; idx2++;
    }
    return list;
}

// Search for smallest element in a sorted list which is larger than pivot
// Uses binary search since list is sorted.
// Input:
//   list, list_size	- list and its size
//   pivot		- value to search for
// Output:
//   last 	- index of the smallest element that is larger than the pivot
//
int split_list_index (int *list, int list_size, int pivot) {
    int first, last, mid;
    first = 0; last = list_size; mid = (first+last)/2;
    while (first < last) {
	if (list[mid] <= pivot) {
	    first = mid+1; mid = (first+last)/2;
	} else {
	    last = mid; mid = (first+last)/2;
	}
    }
    return last;
}

// Comparison routine for qsort (stdlib.h) which is used to sort local list
// Used as follows to sort elements of list[0 ... list_size-1]:
//
// 	qsort(list, list_size, sizeof(int), compare_int)
//
//
int compare_int(const void *a0, const void *b0) {
    int a = *(int *)a0;
    int b = *(int *)b0;
    if (a < b) {
	return -1;
    } else if (a > b) {
	return 1;
    } else {
	return 0;
    }
}

//------------------------------------------------------------------------------
// Miscellaneous routines
//
// Print local list
//
void print_local_list(int *list, int list_size, int my_id) {
    int j;
    for (j = 0; j < list_size; j++) {
	if ((j % 8) == 0) printf("[Proc: %0d]", my_id);
	printf(" %8d", list[j]);
	if ((j % 8) == 7) printf("\n");
    }
    printf("\n");
    return;
}

// Print list: processes print local lists in order of their ranks (from 0 to p-1)
//
void print_list(int *list, int list_size, int my_id, int num_procs) {
    int tag = 0;
    int dummy = 0;
    MPI_Status status;
    if (my_id-1 >= 0) {
	MPI_Recv(&dummy, 1, MPI_INT, my_id-1, tag, MPI_COMM_WORLD, &status);
    }
    print_local_list(list, list_size, my_id);
    if (my_id+1 < num_procs) {
	MPI_Send(&dummy, 1, MPI_INT, my_id+1, tag, MPI_COMM_WORLD);
    }
}

//------------------------------------------------------------------------------
// Main program
//
int main(int argc, char *argv[])
{
    // Local Variables ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    int *list;			// Local list
    int list_size;		// Local list size
    int list_size0;		// Size of initial local list (before sorting)
    int type;			// Method for initializing local lists

    int dim;			// Hypercube dimension
    int k; 			// Sub hypercube dimension
    int nbr_k; 			// Neighbor of this process along dim-k

    int local_median;		// Median of local list
    int pivot;			// Value used to split local list
    int idx;			// index where local list is split
    int list_size_leq;		// Number of elements <= pivot
    int list_size_gt;		// Number of elements > pivot
    int * nbr_list; 		// Sublist received from nbr process
    int nbr_list_size;		// nbr_list size
    int * new_list; 		// List obtained by merging nbr list with
    				// local sublist

    int error, mask, i;		// Miscellaneous work variables

    // MPI variables
    int num_procs; 		// Number of MPI processes
    int my_id;			// Rank/id of this process
    int tag = 0;
    MPI_Status status;

    // Hypercube group and communicator variables to facilitate pivot computation
    // An MPI Group and an MPI communicator are created for hypercube of
    // dimension k; this hypercube includes all processes with ranks
    // that differ from this process in the lowest k bits only
    //
    int sub_hypercube_size; 		// Number of processors in dim-k hypercube
    int * sub_hypercube_processors;	// List of processors in dim-k hypercube
    MPI_Group sub_hypercube_group; 	// Group of processes in dim-k hypercube
    MPI_Comm sub_hypercube_comm;	// Communicator for dim-k hypercube
    MPI_Group hypercube_group;		// Group of all processes

    // Timing variables
    double start, total_time;

    // Hypercube Quicksort Algorithm +++++++++++++++++++++++++++++++++++++++++++

    MPI_Init(&argc,&argv);			// Initialize MPI
    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);	// num_procs = number of MPI processes
    MPI_Comm_rank(MPI_COMM_WORLD,&my_id);	// my_id = rank of this process

    //  Check inputs
    if (argc != 3)  {
	if (my_id == 0)
	    printf("Usage: mpirun -np <number_of_processes> <executable_name> <list_size_per_process> <type>\n");
	exit(0);
    }
    list_size = atoi(argv[argc-2]);
    list_size0 = list_size;		// Save initial list size for reporting at the end
    if ((list_size <= 0) || (list_size > MAX_LIST_SIZE_PER_PROC)) {
	if (my_id == 0)
	    printf("List size outside range [%d ... %d]. Aborting ...\n", 1, MAX_LIST_SIZE_PER_PROC);
	exit(0);
    };
    type = atoi(argv[argc-1]);

    // Compute hypercube dimension: 2^dim = num_procs
    dim = (int) log2(num_procs);
    if (num_procs != (int) pow(2,dim)) {
	if (my_id == 0)
	    printf("Number of processors must be power of 2. Aborting ...\n");
	exit(0);
    }

    // Initialize local list
    list = initialize_list(list_size, type, my_id, num_procs);

    if (VERBOSE > 2) {
	print_list(list, list_size, my_id, num_procs);
    }

    // Start Hypercube Quicksort ..............................................
    start = MPI_Wtime();

    // Sort local list
    qsort(list, list_size, sizeof(int), compare_int);

    // Initialize processor group for hypercube
    MPI_Comm_group(MPI_COMM_WORLD, &hypercube_group);

    // Hypercube Quicksort
    for (k = dim; k > 0; k--) {

        // Find processes that make up the sub-hypercube of dimension k and
	// include this process; the sub-hypercube includes all processes with
	// ranks that differ from this process in the lowest k bits only
	sub_hypercube_size = (int) pow(2,k); mask = (~0) << k;
	sub_hypercube_processors = (int *) calloc(sub_hypercube_size, sizeof(int));
	sub_hypercube_processors[0] = my_id & mask;
	for (i = 1; i < sub_hypercube_size; i++) {
	    sub_hypercube_processors[i] = sub_hypercube_processors[i-1]+1;
	}

	// Construct processor group for sub-hypercube
	MPI_Group_incl(hypercube_group, sub_hypercube_size, sub_hypercube_processors, &sub_hypercube_group);

	// Construct processor communicator to simplify computation of pivot
	// via MPI_Allreduce within the sub-hypercube
	MPI_Comm_create(MPI_COMM_WORLD, sub_hypercube_group, &sub_hypercube_comm);

	// Find median of sorted local list
	local_median = list[list_size/2];

	// MPI-1: Compute pivot for hypercube of dimension k (pivot = mean of medians)
	// MPI_Allreduce can be used with the MPI Communicator sub_hypercube_comm to
	// compute the sum of local_median values on processes of this hypercube

	// ***** Add MPI call here *****
    MPI_Allreduce(&local_median, &pivot, 1, MPI_INT, MPI_SUM, sub_hypercube_comm);

	pivot = pivot/sub_hypercube_size;

	// Search for smallest element in list which is larger than pivot
	// Upon return:
	//   list[0 ... idx-1] <= pivot
	//   list[idx ... list_size-1] > pivot
	idx = split_list_index(list, list_size, pivot);

	list_size_leq = idx;
	list_size_gt = list_size - idx;

	// Communicate with neighbor along dimension k
	nbr_k = neighbor_along_dim_k(my_id, k);

	if (nbr_k > my_id) {
	    // MPI-2: Send number of elements greater than pivot
	    // ***** Add MPI call here *****
        MPI_Send(&list_size_gt, 1, MPI_INT, nbr_k, 0, MPI_COMM_WORLD);

	    // MPI-3: Receive number of elements less than or equal to pivot

	    // ***** Add MPI call here *****
        MPI_Recv(&nbr_list_size, 1, MPI_INT, nbr_k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    // Allocate storage for neighbor's list
	    nbr_list = (int *) calloc(nbr_list_size, sizeof(int));

	    // MPI-4: Send list[idx ... list_size-1] to neighbor

	    // ***** Add MPI call here *****
        MPI_Send(&list[idx], list_size_gt, MPI_INT, nbr_k, 0, MPI_COMM_WORLD);

	    // MPI-5: Receive neighbor's list of elements that are less than or equal to pivot

	    // ***** Add MPI call here *****
        MPI_Recv(nbr_list, nbr_list_size, MPI_INT, nbr_k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    // Merge local list of elements less than or equal to pivot with neighbor's list
	    new_list = merged_list(list, idx, nbr_list, nbr_list_size);

	    // Replace local list with new_list, update size
	    free(list); free(nbr_list);
	    list = new_list;
	    list_size = list_size_leq+nbr_list_size;

	} else {
	    // MPI-6: Receive number of elements greater than pivot
	    // ***** Add MPI call here *****
        MPI_Recv(&nbr_list_size, 1, MPI_INT, nbr_k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    // MPI-7: Send number of elements less than or equal to pivot

	    // ***** Add MPI call here *****
        MPI_Send(&list_size_leq, 1, MPI_INT, nbr_k, 0, MPI_COMM_WORLD);

	    // Allocate storage for neighbor's list
	    nbr_list = (int *) calloc(nbr_list_size, sizeof(int));

	    // MPI-8: Receive neighbor's list of elements that are greater than the pivot

	    // ***** Add MPI call here *****
        MPI_Recv(nbr_list, nbr_list_size, MPI_INT, nbr_k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    // MPI-9: Send list[0 ... idx-1] to neighbor

	    // ***** Add MPI call here *****
        MPI_Send(list, list_size_leq, MPI_INT, nbr_k, 0, MPI_COMM_WORLD);


	    // Merge local list of elements greater than pivot with neighbor's list
	    new_list = merged_list(&list[idx], list_size_gt, nbr_list, nbr_list_size);

	    // Replace local list with new_list, update size
	    free(list); free(nbr_list);
	    list = new_list;
	    list_size = list_size_gt+nbr_list_size;
	}
	// Deallocate processor group, processor communicator,
	// sub_hypercube_processors array; these variables will be
	// reused in the next iteration of this for loop for a hypercube of
	// dimension (k-1)
	MPI_Group_free(&sub_hypercube_group);
	MPI_Comm_free(&sub_hypercube_comm);
	free(sub_hypercube_processors);
    }

    total_time = MPI_Wtime()-start;
    // End Hypercube Quicksort ..............................................

    if (my_id == 0) {
	printf("[Proc: %0d] number of processes = %d, initial local list size = %d, hypercube quicksort time = %f\n", my_id, num_procs, list_size0, total_time);
    }

    // Check if list has been sorted correctly
    check_list(list, list_size, my_id, num_procs);

    if (VERBOSE > 2) {
	print_list(list, list_size, my_id, num_procs);
    }

    MPI_Finalize();				// Finalize MPI
}
