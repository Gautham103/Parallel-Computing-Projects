#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//#include <omp.h>

#define T_PARAM 0.5
#define GET_RAND 0.02 * (((double)rand()/(10.0 * (double)RAND_MAX)) - 0.05)
#define  SIZE 10000
#define  L_PARAM_LENGTH 20
#define PI 3.1415
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

void shuffle(int * arr, int n)
{
  for (int i = 0; i < n; i++)
  {
         int r = rand() % n;
         int temp =  arr[i];
         arr[i] =  arr[r];
         arr[r] = temp;
  }
}

void compute_f_matrix (double **f, node_coord *grid_points, int n, double l1, double l2)
{
    double temp_x, temp_y;

    for (int i = 0; i < n; i++)
    {
        temp_x = ((grid_points[i].x - 0.25) * (grid_points[i].x - 0.25)) / (l1 * l1);
        temp_y = ((grid_points[i].y - 0.25) * (grid_points[i].y - 0.25)) / (l2 * l2);
        f[i][0] = exp(-0.5 * (temp_x + temp_y)) + (grid_points[i].x * 0.2) + (grid_points[i].y * 0.1) + GET_RAND;
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
            K[i][j] = (1/sqrt(2*PI)) * exp(-0.5 * (temp_x + temp_y));
        }
    }
}

void extract_K_matrix (double **output, double **K0, int n, int *train_index)
{
    int ntest = ceil (0.1 * n);
    int ntrain = n - ntest;

    int k = 0, l = 0;
    for (int i = 0; i < ntrain; i++)
    {
        for (int j = 0; j < ntrain; j++)
        {
            output[k][l++] = K0[train_index[i]][train_index[j]];
        }
        k++;
        l = 0;
    }
}

void extract_k_matrix (double **output, double **K0, int n, int *train_index, int *test_index)
{
    int ntest = ceil (0.1 * n);
    int ntrain = n - ntest;
    int k = 0, l = 0;
    for (int i = 0; i < ntrain; i++)
    {
        for (int j = 0; j < ntest; j++)
        {
            output[k][l++] = K0[train_index[i]][test_index[j]];
        }
        k++;
        l = 0;
    }
}

void extract_ftrain_matrix (double **output, double **f, int *index, int ntrain)
{
    int k = 0;
    for (int i = 0; i < ntrain; i++)
    {
        output[0][k++] = f[index[i]][0];
    }
}

void extract_ftest_matrix (double **output, double **f, int *index, int ntest, int n)
{
    int k = 0;
    for (int i = 0; i < ntest; i++)
    {
        output[0][k++] = f[index[i]][0];
    }
}

void calculate_error (double **error, double **f_test, double **f_predicted, int n)
{
    for (int i = 0; i < n; i++)
    {
        error[0][i] = f_test[0][i] - f_predicted[0][i];
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

void get_transpose (double ** output, double **input, int n)
{

    for(int i = 0; i < n; i++)
    {
        output[i][0] = input[0][i];
    }
}

double** run_solver (double **K_inverse, double **k, double **f_train, int train, int test)
{
    double **product = alloc_multidim_matrix (train , test);
//#pragma omp task default (none) shared(product, K_inverse, k, n)
    multiply_matrix (product, K_inverse, k, train , train, train, test);

    double **output = alloc_multidim_matrix (1, test);
//#pragma omp task default (none) shared(output, observed_data_points, product, n)
    multiply_matrix (output, f_train, product, 1, train, train, test);

    return output;

}

void findMin(double** mse_matrix, int n, int *LParamIndex1, int *LParamIndex2)
{
    double min= mse_matrix[0][0];
    int i,j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            if(mse_matrix[i][j] < min)
            {
                min= mse_matrix[i][j];
                *LParamIndex1 = i;
                *LParamIndex2 = j;
            }
        }
}

double ** GPR (double **K0, node_coord *grid_points, double LParam1, double LParam2,
        int train_start, int train_end, int test_start, int test_end, double **K, double ** k,
        double ** K_inverse, double ** f_train, int n, int *train_index, int *test_index)
{
    int train = train_end - train_start + 1;
    int test = test_end - test_start + 1;

    compute_K_matrix (K0, grid_points, n, (double)LParam1, (double)LParam2);
    extract_K_matrix (K , K0, n, train_index);
    for (int i = 0; i < train; i++)
        K[i][i] += T_PARAM;
    extract_k_matrix (k , K0, n, train_index, test_index);
    get_inverse_by_cholesky(train, K, K_inverse);
    return run_solver (K_inverse, k, f_train, train, test);
}

int main(int argc, char* argv[])
{
    int m;
    if(argc != 2)
    {
        printf("Invalid number of input arguements. Please execute the binary as ./a.out m\n");
        return 0;
    }
    else
    {
        m = atoi(argv[1]);
        printf("Matrix dimension = %d \n", m);
    }

    double h = 1.0 / (double)(m + 1);
    double rate = 0.1;
    int n = m * m;
    int ntest = ceil (rate * n);
    int ntrain = n - ntest;

    int train_start = ntest;
    int train_end = n - 1;
    int train = train_end - train_start + 1;

    int test_start = 0;
    int test_end = ntest - 1;
    int test = test_end - test_start + 1;

    double **K0 = alloc_multidim_matrix (n, n);
    double **K = alloc_multidim_matrix (train, train);
    double **k = alloc_multidim_matrix (train, test);
    double **f = alloc_multidim_matrix (n, n);
    double **f_train = alloc_multidim_matrix (train, train);
    double **f_test = alloc_multidim_matrix (1, test);
    double **K_inverse = alloc_multidim_matrix (train, train);
    double** f_predicted;
    double **error = alloc_multidim_matrix (1, test);
    double **error_trans = alloc_multidim_matrix (test, 1);
    double **mse = alloc_multidim_matrix (1, 1);
    double **mse_matrix = alloc_multidim_matrix (L_PARAM_LENGTH, L_PARAM_LENGTH);

    int LParamIndex1 = 0,LParamIndex2 = 0;

    double LParam[L_PARAM_LENGTH] = {0.2500,0.7500,1.2500,1.7500,2.2500,2.7500,3.2500,3.7500,4.2500,
        4.7500,5.2500,5.7500,6.2500,6.7500,7.2500,7.7500,8.2500,8.7500,9.2500,9.7500};


    n = m*m;
    h = 1.0 / (double)(m + 1);

    for (int i = 0; i < 20; i++)
    {
        LParam[i] = (double)LParam[i] / m;
    }

    node_coord *grid_points = (node_coord *) malloc (n * sizeof (node_coord));
    //#pragma omp task default (none) shared(grid_points, m, h)
    initialize_points (grid_points, m, h);
    //#pragma omp taskwait

    int *arr = (int *) malloc (sizeof (int) * n);
    int index = 0;
    int *train_index = (int *) malloc (sizeof (int) * ntrain);
    int *test_index = (int *) malloc (sizeof (int) * ntest);

    for (int i = 0; i < n; i++)
        arr[i] = i;

    shuffle (arr, n);

    for (int i = 0; i < ntrain; i++)
        train_index[index++] = arr[i];
    index = 0;
    for (int i = ntrain; i < n; i++)
        test_index[index++] = arr[i];

    compute_f_matrix (f, grid_points, n, (double) 2/m, (double) 2/m);

    extract_ftrain_matrix (f_train , f, train_index, ntrain);
    extract_ftest_matrix (f_test , f, test_index, ntest, n);

    for (int i = 0; i < L_PARAM_LENGTH; i++)
    {
        for (int j = 0; j < L_PARAM_LENGTH; j++)
        {
            f_predicted = GPR (K0, grid_points, (double)LParam[i], (double)LParam[j],
                    train_start, train_end, test_start, test_end, K, k, K_inverse, f_train, n, train_index, test_index);
            calculate_error (error, f_test, f_predicted, test);
            get_transpose (error_trans, error, test);
            multiply_matrix (mse, error, error_trans, 1 , test, test, 1);
            mse_matrix[i][j] = mse[0][0]/ntest;
            //printf("Finished (l1,l2) = %4.4lf, %4.4lf, mse = %e\n", LParam[i], LParam[j], mse_matrix[i][j]);
            mse[0][0] = 0;
        }
    }

    findMin(mse_matrix, L_PARAM_LENGTH , &LParamIndex1, &LParamIndex2);
    printf ("Initial value l1 = %4.4lf l2 = %4.4lf\n", (double)2/m, (double)2/m);
    printf ("Predicted l1 = %4.4lf l2 = %4.4lf\n", LParam[LParamIndex1], LParam[LParamIndex2]);

    return 0;
}
