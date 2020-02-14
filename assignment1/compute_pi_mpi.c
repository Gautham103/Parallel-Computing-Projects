//
// Value of pi up to 100 decimal places (wikipedia.org):
// 3.14159 26535 89793 23846 26433 83279 50288 41971 69399 37510 
//   58209 74944 59230 78164 06286 20899 86280 34825 34211 70679
//
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
int main(int argc, char *argv[])
{
    int done = 0;
    int i, n, myid, numprocs;
    double mypi, pi, h, sum, x, a, error_pi;

    // Timing variables
    double start, total_time;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    if (myid == 0) { 
	if (argc == 1) { 
	    printf("Enter number of intervals:");
	    scanf("%d",&n);
	} else {
	    n = atoi(argv[1]); 
//	    printf("Number of intervals: %d\n", n);
	}
    }
    start = MPI_Wtime();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    h   = 1.0 / n;
    sum = 0.0;
    for (i = myid + 1; i <= n; i += numprocs) { 
	x = h * (i - 0.5);
	sum += 4.0 / (1.0 + x*x);
    }
    mypi = h * sum;
    MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    total_time = MPI_Wtime()-start;

    if (myid == 0) {
        error_pi = fabs(3.14159265358979323846 - pi)/3.14159265358979323846;
        printf("n = %d, p = %d, pi = %.16f, relative error = %.2e, time (sec) = %8.4f\n", n, numprocs, pi, error_pi, total_time);
    }
    MPI_Finalize();
}
