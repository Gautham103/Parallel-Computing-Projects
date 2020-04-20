#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>


#define NOISE 0.01
#define GET_RAND ((double)rand()/(10.0 * (double)RAND_MAX)) - 0.05
#define  SIZE 10000
typedef double VECTOR[SIZE];
__device__ int n;
__device__ int count;

int BLOCK_SIZE = 16;
__device__ struct node_coord {
    double x;
    double y;
}node_coord;

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


void printK(double *K, int m){
    double *host_K = (double*) malloc(m * m * sizeof(double));
    cudaMemcpy(host_K, K, m * m * sizeof(double), cudaMemcpyDeviceToHost);
    printf("\n Device K ---------------------------------- \n");
    int id = 0;
    for(int i=0; i<m; i++){
        for(int j=0; j<m; j++){
            printf("(%5.5lf) ", host_K[id]);
            id++;
        }
        printf("\n");
    }
}

__global__ void gpu_matrix_mult(double *a,double *b, double *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

int _ConvertSMVer2Cores(int major, int minor) {

    typedef struct {
        int SM; 
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        {   -1, -1 }
    };
    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    return nGpuArchCoresPerSM[index-1].Cores;
}

int getNumThreads(){
    int num_threads = 0;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; device++) 
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        int curr_t =  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
        if(curr_t > num_threads)
            num_threads = curr_t;
    }
    return(num_threads);
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


void print_1d_matrix (char *msg, double* matrix, int n)
{
    printf("\n --------------------- %s --------------------- \n", msg);
    int id = 0;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            printf("%4.9lf ", matrix[id++]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
}

void print_points (struct node_coord *grid_points, int m)
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
void initialize_points (struct node_coord *grid_points, int m, double h)
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

void calculate_observed_data (double **observed_data_points, struct node_coord *grid_points, int n)
{
    for(int id = 0; id < n; id++)
    {
        observed_data_points[0][id] = 1 - (((grid_points[id].x - 0.5) * (grid_points[id].x - 0.5)) +
                       ((grid_points[id].y - 0.5) * (grid_points[id].y - 0.5))) + GET_RAND;
    }

}


__global__ void compute_K_matrix_gpu (int num_threads, double *K, struct node_coord *grid_points, int m)
{
    n = m * m;
    double temp_x, temp_y;

    for(int i = threadIdx.x; i < n; i+= num_threads)
    {
        for (int j = 0; j < n; j++)
        {
            temp_x = (grid_points[i].x - grid_points[j].x) * (grid_points[i].x - grid_points[j].x);
            temp_y = (grid_points[i].y - grid_points[j].y) * (grid_points[i].y - grid_points[j].y);
            K[i*n + j] = exp(-1 * (temp_x + temp_y));
            if (i == j)
                K[i*n + j] += NOISE;
        }
    }
}


__device__ void get_total_sum(double *partial_sum, int dummy) {
    if(threadIdx.x == 0) {
        count = dummy;
        if(count % 2 != 0)
            count++;
        for(int i = 1; i < count; i++)
            partial_sum[0] += partial_sum[i];
    }
}

__global__ void device_run_chol(int num_threads, double *K, double *L, double *partial_sum, int n) {
    for(int k = 0; k < n; k++) {
        for(int i = 0; i < n; i++) {
            if(i == k) {
                partial_sum[threadIdx.x] = 0;
                for(int j = threadIdx.x; j < k; j += num_threads) {
                    partial_sum[threadIdx.x] = partial_sum[threadIdx.x] + pow((L[k * n + j]), 2);
                }
                __syncthreads();
                get_total_sum(partial_sum, (num_threads<k)?num_threads:k);
                if(threadIdx.x == 0) {
                    L[i * n + i] = sqrt(K[k * n + k] - partial_sum[0]);
                }
            } else if( i > k) {
                partial_sum[threadIdx.x] = 0;
                for(int j = threadIdx.x; j < k; j += num_threads) {
                    partial_sum[threadIdx.x] = partial_sum[threadIdx.x] + L[i * n + j] * L[k * n + j];
                }
                __syncthreads();
                get_total_sum(partial_sum, (num_threads<k)?num_threads:k);
                if(threadIdx.x == 0) {
                    L[i * n + k] = K[i * n + k] - partial_sum[0];
                    L[i * n + k] /= L[k * n + k];
                }
            } else {
                if(threadIdx.x == 0)
                    L[i * n + k] = 0;
            }
            __syncthreads();
        }
    }
}


void compute_matrix_part_1 (double** matrix, int row_beg, int col_beg, int row_end, int col_end, int mid, int n)
{
    double *res1 = (double*) malloc((mid + 1) * sizeof(double));
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

    compute_inverse(matrix, row_beg, col_beg, row_beg + mid, col_beg + mid);
    compute_inverse(matrix, row_beg + mid + 1, col_beg + mid + 1, row_end, col_end);
    compute_matrix_part_1 (matrix, row_beg, col_beg, row_end, col_end, mid, n);
    compute_matrix_part_2 (matrix, row_beg, col_beg, row_end, col_end, mid, n);
}


void compute_k (double **k, struct node_coord *rstar, struct node_coord *grid_points, int n)
{
    double temp_x, temp_y;

    for (int i = 0; i < n; i++)
    {
        temp_x = (grid_points[i].x - rstar[0].x) * (grid_points[i].x - rstar[0].x);
        temp_y = (grid_points[i].y - rstar[0].y) * (grid_points[i].y - rstar[0].y);
        k[0][i] = exp(-1 * (temp_x + temp_y));
    }
}

void convert_array (double ** host_K, double * matrix, int n, bool convert_2d)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if (convert_2d)
                host_K[i][j] = matrix[i*n + j];
            else
                matrix[i*n + j] = host_K[i][j];

        }
    }
}

void get_transpose (double ** host_L, double **host_U, int n)
{

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            host_U[j][i] = host_L[i][j];
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

double run_solver (double *L, double *k_out, double *host_obs_data, int n, int num_threads)
{
    unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    double **host_L = alloc_mem_matrix (n);
    double **host_U = alloc_mem_matrix (n);
    double **host_U_trans = alloc_mem_matrix (n);
    double *L_host = (double*) malloc(n * n *sizeof(double));
    cudaMemcpy(L_host, L, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    convert_array(host_L, L_host, n, true);
    get_transpose (host_L, host_U, n);

    compute_inverse(host_U, 0, 0, n - 1, n - 1);

    duplicate_matrix (host_U, host_U_trans, n);
    get_transpose (host_U, host_U_trans, n);
    double *host_inv = (double*) malloc(n * n *sizeof(double));
    convert_array(host_U, host_inv, n, false);
    double *host_inv_trans = (double*) malloc(n * n *sizeof(double));
    convert_array(host_U_trans, host_inv_trans, n, false);


    double *device_k_out, *device_host_inv, *device_host_inv_trans, *device_obs_data;
    cudaMalloc(&device_k_out, (n * n * sizeof(double)));
    cudaMalloc(&device_host_inv, (n * n * sizeof(double)));
    cudaMemcpy(device_k_out, k_out, n* n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_host_inv, host_inv, n* n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&device_host_inv_trans, (n * n * sizeof(double)));
    cudaMalloc(&device_obs_data, (n * n * sizeof(double)));
    cudaMemcpy(device_host_inv_trans, host_inv_trans, n* n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_obs_data, host_obs_data, n* n * sizeof(double), cudaMemcpyHostToDevice);

    double *product1;
    cudaMalloc(&product1, (n * n * sizeof(double)));
    gpu_matrix_mult<<<1, num_threads>>> (device_k_out, device_host_inv,product1, n, n , n);

    double *h_a, *h_b, *h_c;
    cudaMallocHost((void **) &h_a, sizeof(double)*n*n);
    cudaMallocHost((void **) &h_b, sizeof(double)*n*1);
    cudaMallocHost((void **) &h_c, sizeof(double)*n*1);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = host_inv_trans[i*n + j];
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 1; ++j) {
            h_b[i * 1 + j] = host_obs_data[i*1 + j];
        }
    }

    double *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(double)*n*n);
    cudaMalloc((void **) &d_b, sizeof(double)*n*1);
    cudaMalloc((void **) &d_c, sizeof(double)*n*1);

    cudaMemcpy(d_a, h_a, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(double)*n*1, cudaMemcpyHostToDevice);


    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n, n, 1);
    cudaMemcpy(h_c, d_c, sizeof(double)*n*1, cudaMemcpyDeviceToHost);

    double *final_product;
    cudaMalloc(&final_product, (n * n * sizeof(double)));
    
    gpu_matrix_mult<<<dimGrid, dimBlock>>> (product1, d_c, final_product, 1, n , 1);
    double *predicted_value = (double*) malloc(n * n * sizeof(double));
    cudaMemcpy(predicted_value, final_product, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    return predicted_value[0];

}
int main(int argc, char* argv[])
{
    struct node_coord *rstar = (struct node_coord *) malloc (1 * sizeof (struct node_coord));
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

    n = m * m;
    h = 1.0 / (double)(m + 1);
    int num_threads = getNumThreads();
    cudaEvent_t start_overall, end_overall, start_LU, end_LU, start_solver, end_solver;

    struct node_coord *grid_points = (struct node_coord *) malloc (n * sizeof (struct node_coord));
    initialize_points (grid_points, m, h);


    struct node_coord *device_grid_points;
    cudaMalloc(&device_grid_points, (n * sizeof(struct node_coord)));
    cudaMemcpy(device_grid_points, grid_points, n * sizeof(struct node_coord), cudaMemcpyHostToDevice);


    double **observed_data_points = alloc_mem_matrix (n);
    calculate_observed_data (observed_data_points, grid_points, n);
    double *host_obs_data;
    cudaMallocHost((void **) &host_obs_data, sizeof(double)*n*n);
    convert_array (observed_data_points, host_obs_data, n , false);

    double **k = alloc_mem_matrix (n);
    double *k_out = (double*) malloc(n * n *sizeof(double));
    compute_k (k, rstar, grid_points, n);
    convert_array(k, k_out, n, false);

    unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventCreate(&start_overall);
    cudaEventCreate(&end_overall);
    double *K1;
    cudaMalloc(&K1, (n * n * sizeof(double)));
    compute_K_matrix_gpu<<<1, num_threads>>>(num_threads, K1, device_grid_points, m);

    double *L, *partial_sum;
    cudaMalloc(&L, (n * n * sizeof(double)));
    cudaMalloc(&partial_sum, num_threads * sizeof(double));
    cudaEventCreate(&start_LU);
    cudaEventCreate(&end_LU);
    cudaEventRecord(start_LU, 0);
    cudaEventRecord(start_overall, 0);

    device_run_chol<<<1,num_threads>>>(num_threads, K1, L, partial_sum, n);
    cudaEventRecord(end_LU, 0);
    cudaEventSynchronize(end_LU);


    cudaEventCreate(&start_solver);
    cudaEventCreate(&end_solver);
    cudaEventRecord(start_solver, 0);
    double predicted_value = run_solver (L, k_out, host_obs_data, n, num_threads);
    cudaEventRecord(end_solver, 0);
    cudaEventSynchronize(end_solver);
    cudaEventRecord(end_overall, 0);
    cudaEventSynchronize(end_overall);

    printf ("Predicted value is %lf\n", predicted_value);

    double actual_value = 1 - (((rstar[0].x - 0.5) * (rstar[0].x - 0.5)) +
            ((rstar[0].y - 0.5) * (rstar[0].y - 0.5))) + GET_RAND;

    printf ("Actual value is %lf\n", actual_value);
    float time_overall, time_LU, time_solver;
    cudaEventElapsedTime(&time_overall, start_overall, end_overall);
    cudaEventElapsedTime(&time_LU, start_LU, end_LU);
    cudaEventElapsedTime(&time_solver, start_solver, end_solver);


    printf("Time Taken by Cholesky = %f ms\n", time_LU);
    printf("Time Taken by Solver = %f ms\n", time_solver);
    printf("Overall Time Taken = %f ms\n", time_overall);
    return 0;
}

