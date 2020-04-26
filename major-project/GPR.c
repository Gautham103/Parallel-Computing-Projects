#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//#include <omp.h>

#define NOISE 0.01
#define GET_RAND 0.02 * (((double)rand()/(10.0 * (double)RAND_MAX)) - 0.05)
#define  SIZE 10000
#define  L_PARAM_LENGTH 20
typedef double VECTOR[SIZE];

typedef struct node {
    double x;
    double y;
} node_coord;

double** alloc_multidim_matrix (int row, int col)
{
    double **matrix = NULL;
    matrix = (double**) calloc (row, sizeof(double*));
    for(int i = 0; i < row; i++)
    {
        matrix[i] = (double*) calloc (col, sizeof(double));
    }
    return matrix;
}


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
//#pragma omp for
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

void print_multidem_matrix (char *msg, double** matrix, int row, int col)
{
    printf("\n --------------------- %s --------------------- \n", msg);
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            printf("%4.4lf ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
}

void print_matrix (char *msg, double** matrix, int n)
{
    printf("\n --------------------- %s --------------------- \n", msg);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            printf("%4.4lf ", matrix[i][j]);
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
//#pragma omp for
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

void compute_f_matrix (double **f, node_coord *grid_points, int n, double l1, double l2)
{
    double temp_x, temp_y;

    for (int i = 0; i < n; i++)
    {
        temp_x = ((grid_points[i].x - 0.25) * (grid_points[i].x - 0.25)) / (l1 * l1);
        temp_y = ((grid_points[i].y - 0.25) * (grid_points[i].y - 0.25)) / (l2 * l2);
        f[i][0] = exp(-0.5 * (temp_x + temp_y)) + (grid_points[i].x * 0.2) + (grid_points[i].y * 0.1);// + GET_RAND;
        //if (i == j)
            //f[i][0] += NOISE;
    }

}




void compute_K_matrix (double **K, node_coord *grid_points, int n, double l1, double l2)
{
    double temp_x, temp_y;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            temp_x = ((grid_points[i].x - grid_points[j].x) * (grid_points[i].x - grid_points[j].x)) / (l1 * l1);
            temp_y = ((grid_points[i].y - grid_points[j].y) * (grid_points[i].y - grid_points[j].y)) / (l2 * l2);
            K[i][j] = exp(-0.5 * (temp_x + temp_y));
        }
    }
}

void extract_matrix (double **output, double **K0, int row_beg, int row_end, int col_beg, int col_end)
{
    int k = 0, l = 0;
    //printf ("row_beg %d row_end %d col_beg %d col_end %d\n", row_beg, row_end, col_beg, col_end);
    for (int i = row_beg; i <= row_end; i++)
    {
        for (int j = col_beg; j <= col_end; j++)
        {
            output[k][l++] = K0[i][j];
        }
        k++;
        l = 0;
    }
}

void extract_f_matrix (double **output, double **f, int row_beg, int row_end)
{
    int k = 0;
    for (int i = row_beg; i <= row_end; i++)
    {
        output[k++][0] = f[i][0];
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

//#pragma omp for
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

//#pragma omp for
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
//#pragma omp for
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
//#pragma omp task default (none) shared(product, K_inverse, k, n)
    multiply_matrix (product, K_inverse, k, n , n, n, 1);

//#pragma omp task default (none) shared(output, observed_data_points, product, n)
    multiply_matrix (output, observed_data_points, product, 1, n, n, 1);

    return output[0][0];

}

int main(int argc, char* argv[])
{
    node_coord *rstar = (node_coord *) malloc (1 * sizeof (node_coord));
    int m;
    int n;
    double h;
    int ntest, ntrain;
    double LParam[L_PARAM_LENGTH] = {0.2500,0.7500,1.2500,1.7500,2.2500,2.7500,3.2500,3.7500,4.2500,
        4.7500,5.2500,5.7500,6.2500,6.7500,7.2500,7.7500,8.2500,8.7500,9.2500,9.7500};
    double **mse_matrix = alloc_mem_matrix (L_PARAM_LENGTH);
    double rate = 0.1;

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

    for (int i = 0; i < 20; i++)
    {
        LParam[i] = LParam[i] / m;
    }

    node_coord *grid_points = (node_coord *) malloc (n * sizeof (node_coord));
//#pragma omp task default (none) shared(grid_points, m, h)
    initialize_points (grid_points, m, h);
//#pragma omp taskwait

    double **observed_data_points = alloc_mem_matrix (n);
    calculate_observed_data (observed_data_points, grid_points, n);
    print_points (grid_points, m);

    double **K0 = alloc_mem_matrix (n);
    compute_K_matrix (K0, grid_points, n, (double)LParam[19], (double)LParam[19]);
    print_matrix ("K0 Matrix", K0, n);

    ntest = ceil (rate * n);
    ntrain = n - ntest;
    int train_start = ntest;
    int train_end = n - 1;

    int test_start = 0;
    int test_end = ntest - 1;

    double **K = alloc_multidim_matrix ((train_end - train_start + 1), (train_end - train_start + 1));
    extract_matrix (K , K0, train_start, train_end, train_start, train_end);
    //print_multidem_matrix ("K Matrix", K, (train_end - train_start + 1), (train_end - train_start + 1));

    double **k = alloc_multidim_matrix ((train_end - train_start + 1), (test_end - test_start + 1));
    extract_matrix (k , K0, train_start, train_end, test_start, test_end);
    //print_multidem_matrix ("k Matrix", K, (train_end - train_start + 1), (test_end - test_start + 1));

    double **f = alloc_mem_matrix (n);
    compute_f_matrix (f, grid_points, n, (double) 2/m, (double) 2/m);
    //print_matrix ("f Matrix", f, n );

    double **f_updated = alloc_multidim_matrix ((train_end - train_start + 1), (train_end - train_start + 1));
    extract_f_matrix (f_updated , f, train_start, train_end);
    //print_multidem_matrix ("f_updated Matrix", f_updated, (train_end - train_start + 1), (train_end - train_start + 1));


#if 0
    double **k = alloc_mem_matrix (n);
    compute_k (k, rstar, grid_points, n);

    double **K_inverse = alloc_mem_matrix (n);
    //double start_time = omp_get_wtime();
//#pragma omp task default (none) shared(n, K, K_inverse)
    get_inverse_by_cholesky(n, K, K_inverse);
//#pragma omp taskwait
    //double end_time = omp_get_wtime();
    //printf ("Time taken by cholesky = %10.5e\n", end_time - start_time);

   // start_time = omp_get_wtime();
    double predicted_value = run_solver (K_inverse, k, observed_data_points, n);
    //end_time = omp_get_wtime();

    double actual_value = 1 - (((rstar[0].x - 0.5) * (rstar[0].x - 0.5)) +
                                    ((rstar[0].y - 0.5) * (rstar[0].y - 0.5))) + GET_RAND;
 //   printf ("Time taken by solver = %10.5e\n", end_time - start_time);
    printf ("Actual value is %lf\n", actual_value);
    printf ("Predicted value is %lf\n", predicted_value);
#endif

    return 0;
}
