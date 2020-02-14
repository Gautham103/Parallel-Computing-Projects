#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

double** alloc_mem_matrix (int n)
{
    double **matrix = NULL;
    matrix = (double**) calloc (n, sizeof(double*));
    for(int i = 0; i < n; i++)
    {
        matrix[i] = (double*) calloc (n, sizeof(double));
    }
    return matrix;
}

void print_matrix (double** matrix, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            printf("%4.9lf ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
}

double** multiply_matrix (double** matrix1, double** matrix2, int n)
{
    double** product = alloc_mem_matrix (n);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            for(int k = 0; k < n; k++)
            {
                product[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return product;
}

void compute_matrix_part_1 (double** matrix, int row_beg, int col_beg, int row_end, int col_end, int mid, int n)
{
    double *res1 = (double*) malloc((mid + 1) * sizeof(double));
#pragma omp for
    for(int j = col_beg + mid + 1; j < col_end + 1; ++j)
    {
        for(int i = row_beg; i < row_beg + mid + 1; ++i)
        {
            res1[i - row_beg] = 0;
            for(int k = 0; k < mid + 1; ++k)
            {
                res1[i - row_beg] += -1 * matrix[i][col_beg + k] * matrix[row_beg + k][j];
            }
        }
        for(int l = 0; l < mid + 1; ++l)
        {
            matrix[row_beg + l][j] = res1[l];
        }
    }
}

void compute_matrix_part_2 (double** matrix, int row_beg, int col_beg, int row_end, int col_end, int mid, int n)
{
    double *res2 = (double*) malloc ((n - mid - 1) * sizeof(double));
#pragma omp for
    for(int i = row_beg; i < row_beg + mid + 1; ++i)
    {
        for(int j = col_beg + mid + 1; j < col_beg + n; ++j)
        {
            res2 [j - col_beg - mid - 1] = 0;
            for(int k = 0; k < n - mid - 1; ++k)
            {
                res2[j - col_beg - mid - 1] += matrix[i][col_beg + mid + 1 + k] * matrix[row_beg + mid + 1 + k][j];
            }
        }
        for(int l = 0; l < n - mid - 1; ++l)
        {
            matrix[i][col_beg + mid + 1 + l] = res2[l];
        }
    }
}

void compute_inverse (double** matrix, int row_beg, int col_beg, int row_end, int col_end)
{
    int n = row_end - row_beg + 1;

    if(n == 1)
    {
        matrix[row_beg][col_beg] = 1 / matrix[row_beg][col_beg];
        return;
    }

    int mid = (n - 1) / 2;

#pragma omp parallel
    {
#pragma omp single
        {
#pragma omp task default (none) shared(matrix, row_beg, col_beg, mid)
            compute_inverse(matrix, row_beg, col_beg, row_beg + mid, col_beg + mid);
#pragma omp task default (none) shared(matrix, row_beg, col_beg, row_end, col_end, mid)
            compute_inverse(matrix, row_beg + mid + 1, col_beg + mid + 1, row_end, col_end);
#pragma omp taskwait
        }
    }

#pragma omp parallel default (none) shared(matrix, row_beg, col_beg, row_end, col_end, mid, n)
    {
        compute_matrix_part_1 (matrix, row_beg, col_beg, row_end, col_end, mid, n);
        compute_matrix_part_2 (matrix, row_beg, col_beg, row_end, col_end, mid, n);
    }
}

double** upper_triangular_matrix (double** matrix, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < i; j++)
        {
            matrix[i][j] = 0;
        }
    }

    return matrix;
}

void initialize_matrix (double** matrix, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            matrix[i][j] = rand() % 1000000;
        }
    }
}

void duplicate_matrix (double** input_matrix, double** output_matrix, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            output_matrix[i][j] = input_matrix[i][j];
        }
    }
}

int main (int argc, char** argv)
{
    int n = 0;
    if(argc != 2)
    {
        printf ("Enter matrix size\n");
        exit(-1);
    }

    n = atoi(argv[1]);

    double** input_matrix = alloc_mem_matrix (n);
    double** output_matrix = alloc_mem_matrix (n);

    initialize_matrix (input_matrix, n);
    input_matrix = upper_triangular_matrix (input_matrix, n);
    duplicate_matrix (input_matrix, output_matrix, n);

    printf ("Input matrix \n");
    print_matrix (input_matrix, n);

    double start_time = omp_get_wtime();
    compute_inverse(output_matrix, 0, 0, n - 1, n - 1);
    double end_time = omp_get_wtime();

    printf ("Inverse matrix \n");
    print_matrix (output_matrix, n);

    double** identity_matrix = multiply_matrix(input_matrix, output_matrix, n);
    printf ("Product matrix of R and Rinverse \n");
    print_matrix (identity_matrix, n);

    printf ("compute_inverse time = %10.5e\n", end_time - start_time);
    return 0;
}


