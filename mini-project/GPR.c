#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NOISE_PARAMETER 0.01
#define GET_RAND ((double)rand()/(10.0 * (double)RAND_MAX)) - 0.05

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
void print_matrix (double** matrix, int n)
{
    printf("\n --------------------- Printing K matrix --------------------- \n");
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

void calculate_observed_data (double *observed_data_points, node_coord *grid_points, int n)
{
    for(int id = 0; id < n; id++)
    {
        observed_data_points[id] = 1 - (((grid_points[id].x - 0.5) * (grid_points[id].x - 0.5)) +
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

void compute_k (double *k, node_coord *rstar, node_coord *grid_points, int n)
{
    double temp_x, temp_y;

    for (int i = 0; i < n; i++)
    {
        temp_x = (grid_points[i].x - rstar[0].x) * (grid_points[i].x - rstar[0].x);
        temp_y = (grid_points[i].y - rstar[0].y) * (grid_points[i].y - rstar[0].y);
        k[i] = exp(-1 * (temp_x + temp_y));
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

    double *observed_data_points = (double *) malloc (n * sizeof (double));
    calculate_observed_data (observed_data_points, grid_points, n);
    print_data (observed_data_points, n);

    double **K = alloc_mem_matrix (n);
    compute_K_matrix (K, grid_points, n);
    print_matrix (K, n);

    double *k = (double *) malloc (n * sizeof (double));
    compute_k (k, rstar, grid_points, n);
    print_data (k, n);
    return 0;
}
