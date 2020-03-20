#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define NOISE 0.01
#define GET_RAND ((double)rand()/(10.0 * (double)RAND_MAX)) - 0.05
#define  SIZE 10000
typedef double VECTOR[SIZE];

typedef struct node {
    double x;
    double y;
} node_coord;

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

double** multiply_matrix (double** product, double** matrix1, double** matrix2, int r1, int c1, int r2, int c2)
{
#pragma omp for
    for(int i = 0; i < r1; i++)
    {
        for(int j = 0; j < c2; j++)
        {
            for(int k = 0; k < c1; k++)
            {
                product[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return product;
}

void print_matrix (char *msg, double** matrix, int n)
{
    printf("\n --------------------- %s --------------------- \n", msg);
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

void print_points (node_coord *grid_points, int m)
{
    int id = 0;
    printf("\n --------------------- Printing grid coordinates --------------------- \n");
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < m; j++)
        {
            printf("(%3.5lf %3.5lf) ", grid_points[id].x, grid_points[id].y);
            id++;
        }
        printf("\n");
    }

}

void print_data (double *data_points, int n)
{
    printf("\n --------------------- Printing data points --------------------- \n");
    for(int id = 0; id < n; id++)
    {
        printf ("%3.5lf  ", data_points[id]);
    }
    printf("\n");

}
void initialize_points (node_coord *grid_points, int m, double h)
{
    int id = 0;
#pragma omp for
    for(int i = 1; i <= m; i++)
    {
        for(int j = 1; j <= m; j++)
        {
            grid_points[id].x = i * h;
            grid_points[id].y = j * h;
            id++;
        }
    }
}

void calculate_observed_data (double **observed_data_points, node_coord *grid_points, int n)
{
    for(int id = 0; id < n; id++)
    {
        observed_data_points[0][id] = 1 - (((grid_points[id].x - 0.5) * (grid_points[id].x - 0.5)) +
                       ((grid_points[id].y - 0.5) * (grid_points[id].y - 0.5))) + GET_RAND;
    }

}
void compute_K_matrix (double **K, node_coord *grid_points, int n)
{
    double temp_x, temp_y;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            temp_x = (grid_points[i].x - grid_points[j].x) * (grid_points[i].x - grid_points[j].x);
            temp_y = (grid_points[i].y - grid_points[j].y) * (grid_points[i].y - grid_points[j].y);
            K[i][j] = exp(-1 * (temp_x + temp_y));
            if (i == j)
                K[i][j] += NOISE;
        }
    }
}

void compute_k (double **k, node_coord *rstar, node_coord *grid_points, int n)
{
    double temp_x, temp_y;

    for (int i = 0; i < n; i++)
    {
        temp_x = (grid_points[i].x - rstar[0].x) * (grid_points[i].x - rstar[0].x);
        temp_y = (grid_points[i].y - rstar[0].y) * (grid_points[i].y - rstar[0].y);
        k[i][0] = exp(-1 * (temp_x + temp_y));
    }

}
void cholesky_compute(int n, double ** matrix, VECTOR p)
{
    int i,j,k;
    double sum;

#pragma omp for
    for (i = 0; i < n; i++)
    {
        for (j = i; j < n; j++)
        {
            sum = matrix[i][j];
            for (k = i - 1; k >= 0; k--)
            {
                sum -= matrix[i][k] * matrix[j][k];
            }
            if (i == j)
            {
                if (sum <= 0)
                {
                    printf(" a is not positive definite!\n");
                }
                p[i] = sqrt(sum);
            }
            else
            {
                matrix[j][i] = sum / p[i];
            }
        }
    }
}


void decompose_and_get_inverse (int n, double ** input_matrix, double ** output_matrix)
{
    int i,j,k;
    double sum;
    VECTOR p;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            output_matrix[i][j] = input_matrix[i][j];

    cholesky_compute (n, output_matrix, p);

#pragma omp for
    for (i = 0; i < n; i++)
    {
        output_matrix[i][i] = 1 / p[i];
        for (j = i + 1; j < n; j++)
        {
            sum = 0;
            for (k = i; k < j; k++)
            {
                sum -= output_matrix[j][k] * output_matrix[k][i];
            }
            output_matrix[j][i] = sum / p[j];
        }
    }
}

void get_inverse_by_cholesky (int n, double ** input_matrix, double ** output_matrix)
{
    int i,j,k;
    decompose_and_get_inverse (n, input_matrix, output_matrix);
#pragma omp for
    for (i = 0; i < n; i++)
    {
        for (j = i + 1; j < n; j++)
        {
            output_matrix[i][j] = 0.0;
        }
    }
    for (i = 0; i < n; i++)
    {
        output_matrix[i][i] *= output_matrix[i][i];
        for (k = i + 1; k < n; k++)
        {
            output_matrix[i][i] += output_matrix[k][i] * output_matrix[k][i];
        }
        for (j = i + 1; j < n; j++)
        {
            for (k = j; k < n; k++)
            {
                output_matrix[i][j] += output_matrix[k][i] * output_matrix[k][j];
            }
        }
    }
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < i; j++)
        {
            output_matrix[i][j] = output_matrix[j][i];
        }
    }
}

double run_solver (double **K_inverse, double **k, double **observed_data_points, int n)
{
    double **product = alloc_mem_matrix (n);
    double **output = alloc_mem_matrix (n);
#pragma omp task default (none) shared(product, K_inverse, k, n)
    multiply_matrix (product, K_inverse, k, n , n, n, 1);

#pragma omp task default (none) shared(output, observed_data_points, product, n)
    multiply_matrix (output, observed_data_points, product, 1, n, n, 1);

    return output[0][0];

}

int main(int argc, char* argv[])
{
    node_coord *rstar = (node_coord *) malloc (1 * sizeof (node_coord));
    int m;
    int n;
    double h;

    if(argc != 4)
    {
        printf("Invalid number of input arguements. Please execute the binary as ./a.out m x* y*\n");
        return 0;
    }
    else
    {
        m = atoi(argv[1]);
        rstar[0].x = atof(argv[2]);
        rstar[0].y = atof(argv[3]);
        printf("Size: %d \n", m);
        printf("Given coordinates (%f, %f)\n", rstar[0].x, rstar[0].y);
    }

    n = m*m;
    h = 1.0 / (double)(m + 1);

    node_coord *grid_points = (node_coord *) malloc (n * sizeof (node_coord));
#pragma omp task default (none) shared(grid_points, m, h)
    initialize_points (grid_points, m, h);
#pragma omp taskwait

    double **observed_data_points = alloc_mem_matrix (n);
    calculate_observed_data (observed_data_points, grid_points, n);

    double **K = alloc_mem_matrix (n);
    compute_K_matrix (K, grid_points, n);

    double **k = alloc_mem_matrix (n);
    compute_k (k, rstar, grid_points, n);

    double **K_inverse = alloc_mem_matrix (n);
    double start_time = omp_get_wtime();
#pragma omp task default (none) shared(n, K, K_inverse)
    get_inverse_by_cholesky(n, K, K_inverse);
#pragma omp taskwait
    double end_time = omp_get_wtime();
    printf ("Time taken by cholesky = %10.5e\n", end_time - start_time);

    start_time = omp_get_wtime();
    double predicted_value = run_solver (K_inverse, k, observed_data_points, n);
    end_time = omp_get_wtime();

    double actual_value = 1 - (((rstar[0].x - 0.5) * (rstar[0].x - 0.5)) +
                                    ((rstar[0].y - 0.5) * (rstar[0].y - 0.5))) + GET_RAND;
    printf ("Time taken by solver = %10.5e\n", end_time - start_time);
    printf ("Actual value is %lf\n", actual_value);
    printf ("Predicted value is %lf\n", predicted_value);

    return 0;
}
