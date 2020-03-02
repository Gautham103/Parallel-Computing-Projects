#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NOISE_PARAMETER 0.01
#define GET_RAND ((double)rand()/(10.0 * (double)RAND_MAX)) - 0.05
#define  SIZE 25
typedef double VEC[SIZE];

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
                       ((grid_points[id].y - 0.5) * (grid_points[id].y - 0.5)));
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
                K[i][j] += 0.01;
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
void choldc1(int n, double ** a, VEC p)
{
    int i,j,k;
    double sum;

    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            sum = a[i][j];
            for (k = i - 1; k >= 0; k--) {
                sum -= a[i][k] * a[j][k];
            }
            if (i == j) {
                if (sum <= 0) {
                    printf(" a is not positive definite!\n");
                }
                p[i] = sqrt(sum);
            }
            else {
                a[j][i] = sum / p[i];
            }
        }
    }
}


void choldcsl(int n, double ** A, double ** a)
{
    int i,j,k; double sum;
    VEC p;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            a[i][j] = A[i][j];
    choldc1(n, a, p);
    for (i = 0; i < n; i++) {
        a[i][i] = 1 / p[i];
        for (j = i + 1; j < n; j++) {
            sum = 0;
            for (k = i; k < j; k++) {
                sum -= a[j][k] * a[k][i];
            }
            a[j][i] = sum / p[j];
        }
    }
}

void cholsl(int n, double ** A, double ** a)
{
    int i,j,k;
    choldcsl(n,A,a);
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            a[i][j] = 0.0;
        }
    }
    for (i = 0; i < n; i++) {
        a[i][i] *= a[i][i];
        for (k = i + 1; k < n; k++) {
            a[i][i] += a[k][i] * a[k][i];
        }
        for (j = i + 1; j < n; j++) {
            for (k = j; k < n; k++) {
                a[i][j] += a[k][i] * a[k][j];
            }
        }
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            a[i][j] = a[j][i];
        }
    }
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
    initialize_points (grid_points, m, h);
    print_points (grid_points, m);

    double **observed_data_points = alloc_mem_matrix (n);
    calculate_observed_data (observed_data_points, grid_points, n);
    print_matrix ("observed_data", observed_data_points, n);

    double **K = alloc_mem_matrix (n);
    compute_K_matrix (K, grid_points, n);
    print_matrix ("K matrix", K, n);

    double **k = alloc_mem_matrix (n);
    compute_k (k, rstar, grid_points, n);
    print_matrix ("k vector", k, n);

    double **K_inverse = alloc_mem_matrix (n);
    cholsl(n, K, K_inverse);
    print_matrix ("k inverse", K_inverse, n);

    double **product = alloc_mem_matrix (n);
    multiply_matrix (product, K_inverse, k, n , n, n, 1);
    print_matrix ("product of K_inverse and k vector", product, n);

    double **output = alloc_mem_matrix (n);
    multiply_matrix (output, observed_data_points, product, 1, n, n, 1);
    printf ("Predicted value is %lf\n", output[0][0]);
    return 0;

}
