#include "./static_header.hpp"
__device__ void direct_compute_wd_pow_h(double *result_real, double *result_imag, int d, cuDoubleComplex h)
{
    // Extract real and imaginary parts of h
    double real_h = cuCreal(h);
    double imag_h = cuCimag(h);

    // Compute the natural logarithm of wd, ln(wd) = i * (2π/d)
    // double ln_wd_real = 0.0;          // Real part of ln(wd) is 0
    double ln_wd_imag = 2 * M_PI / d; // Imaginary part of ln(wd)

    // Compute h * ln(wd) (correct complex multiplication)
    double exponent_real = -imag_h * ln_wd_imag; // Real part of h * ln(wd)
    double exponent_imag = real_h * ln_wd_imag;  // Imaginary part of h * ln(wd)

    // Compute e^(h * ln(wd)) = e^(exponent_real + i * exponent_imag)
    double magnitude = exp(exponent_real);         // Magnitude = e^(real part of exponent)
    *result_real = magnitude * cos(exponent_imag); // Real part of result
    *result_imag = magnitude * sin(exponent_imag); // Imaginary part of result
    return;
}
__device__ void inverse_compute_wd_pow_h(double *result_real, double *result_imag, int d, cuDoubleComplex h)
{
    // Extract real and imaginary parts of h
    double real_h = cuCreal(h);
    double imag_h = cuCimag(h);

    // Compute the natural logarithm of wd, ln(wd) = -i * (2π/d) for IFFT
    double ln_wd_imag = -2 * M_PI / d; // Reversed sign for IFFT

    // Compute h * ln(wd) (correct complex multiplication)
    double exponent_real = -imag_h * ln_wd_imag; // Real part of h * ln(wd)
    double exponent_imag = real_h * ln_wd_imag;  // Imaginary part of h * ln(wd)

    // Compute e^(h * ln(wd)) = e^(exponent_real + i * exponent_imag)
    double magnitude = exp(exponent_real);         // Magnitude = e^(real part of exponent)
    *result_real = magnitude * cos(exponent_imag); // Real part of result
    *result_imag = magnitude * sin(exponent_imag); // Imaginary part of result
}

__device__ void thread_write(bool direct, int t_x, int d, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t)
{
    double result_real, result_imag;

    bool up_down; // up = 1 down = 0 ( if down it will write on x otherwise it will write on t)
    int tmp_index;
    int p;
    cuDoubleComplex w;

    if ((t_x % d) < d / 2)
    {
        up_down = 0;
    }
    else
        up_down = 1;

    // DOWN CASE
    if (!up_down)
    {
        tmp_index = t_x + d / 2;
        // printf("Thread n: %d, Iteration : %d => up_down = %d, index_of_writing : %d\n",t_x, j ,up_down, tmp_index);
        x[tmp_index] = y[t_x];
        x[t_x] = y[t_x];
    }
    // UP CASE
    else
    {
        tmp_index = t_x - d / 2;
        // printf("Thread n: %d, Iteration : %d => up_down = %d, index_of_writing : %d\n",t_x, j ,up_down, tmp_index);
        p = (int)(tmp_index % d);
        if (direct)
        {
            direct_compute_wd_pow_h(&result_real, &result_imag, d, make_cuDoubleComplex(static_cast<double>(p), 0.0));
        }
        else
        {
            inverse_compute_wd_pow_h(&result_real, &result_imag, d, make_cuDoubleComplex(static_cast<double>(p), 0.0));
        }
        w = make_cuDoubleComplex(result_real, result_imag);
        t[tmp_index] = cuCmul(y[t_x], w);
        t[t_x] = cuCmul(y[t_x], w);
    }

    __syncthreads();
}

__device__ void thread_sum(int t_x, int d, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t)
{
    if (t_x % d < d / 2)
    {
        y[t_x] = cuCadd(x[t_x], t[t_x]);
    }
    else
    {
        y[t_x] = cuCsub(x[t_x], t[t_x]);
    }
    __syncthreads();
}

__device__ void permutation(int t_x, int threadIdx_x, int log_n, int input_size, cuDoubleComplex *shared_y, cuDoubleComplex *a)
{
    if (t_x >= input_size)
        return;
    int j = 0;
    for (int k = 0; k < log_n; k++)
    {
        if (t_x & (1 << k))
        {
            j |= (1 << (log_n - 1 - k));
        }
    }
    shared_y[threadIdx_x] = a[j];
}

__global__ void parallel_fft_first_computation(bool direct, int *first_computation_j, int gpu_grid_size, int input_size, int input_grid_size, int *atomic_array, cuDoubleComplex *a, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t, int log_n)
{
    unsigned int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ cuDoubleComplex shared_y[THREAD_PER_BLOCK];
    __shared__ cuDoubleComplex shared_x[THREAD_PER_BLOCK];
    __shared__ cuDoubleComplex shared_t[THREAD_PER_BLOCK];
    int d;
    int tmp_j;
    bool flag = false;
    permutation(t_x, threadIdx.x, log_n, input_size, shared_y, a);
    __syncthreads();

    for (int j = 1; j <= log_n && !flag; j++)
    {
        d = 1 << j;
        if (d < THREAD_PER_BLOCK)
        {
            thread_write(direct, threadIdx.x, d, shared_y, shared_x, shared_t);
            thread_sum(threadIdx.x, d, shared_y, shared_x, shared_t);
        }
        else if (d == THREAD_PER_BLOCK)
        {
            thread_write(direct, threadIdx.x, d, shared_y, shared_x, shared_t);
            thread_sum(threadIdx.x, d, shared_y, shared_x, shared_t);
            tmp_j = j;
            flag = true;
        }
        __syncthreads();
    }

    if (t_x == 0)
    {
        first_computation_j[0] = tmp_j;
    }

    y[t_x] = shared_y[threadIdx.x];
    __syncthreads();
    return;
}

__global__ void parallel_fft_second_computation(bool direct, int actual_computation_j, int gpu_grid_size, int input_size, int input_grid_size, int *atomic_array, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t, int log_n)
{

    int d, prec_d;
    unsigned int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int element_thread_computation;

    int num_iteration;

    int j = actual_computation_j;
    prec_d = 1 << (j - 1);
    d = 1 << j;
    num_iteration = d / THREAD_PER_BLOCK;

    if ((blockIdx.x % num_iteration == 0))
    {

        for (int i = 0; i < num_iteration; i++)
        {
            element_thread_computation = t_x + blockDim.x * i;
            thread_write(direct, element_thread_computation, d, y, x, t);
        }

        for (int i = 0; i < num_iteration; i++)
        {
            element_thread_computation = t_x + blockDim.x * i;
            thread_sum(element_thread_computation, d, y, x, t);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&atomic_array[j], 1);
        }
        __syncthreads();
    }
    else
    {

        return;
    }
}

void compute_row_wise_fft(bool direct, int grid_size, int row_size, int log_n, cuDoubleComplex *a, cuDoubleComplex *y, cuDoubleComplex *x, cuDoubleComplex *t, int *atomic_array, int *first_computation_j)
{
    dim3 dimGrid(grid_size);
    dim3 dimBlock(THREAD_PER_BLOCK);
    // Step 1: Allocate an array of streams
    cudaStream_t *streams = (cudaStream_t *)malloc(row_size * sizeof(cudaStream_t));
    for (int i = 0; i < row_size; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < row_size; i++)
    {
        parallel_fft_first_computation<<<dimGrid, dimBlock, 0, streams[i]>>>(direct, first_computation_j + i, grid_size, N, grid_size, atomic_array, a + i * row_size, y + i * row_size, x + i * row_size, t + i * row_size, log_n);
    }
    for (int i = 0; i < row_size; i++)
    {
        cudaStreamSynchronize(streams[i]);
    }

    for (int j = first_computation_j[0] + 1; j <= log_n; j++)
    {

        for (int i = 0; i < row_size; i++)
        {
            parallel_fft_second_computation<<<dimGrid, dimBlock, 0, streams[i]>>>(direct, j, grid_size, N, grid_size, atomic_array, y + i * row_size, x + i * row_size, t + i * row_size, log_n);
        }
        for (int i = 0; i < row_size; i++)
        {
            cudaStreamSynchronize(streams[i]);
        }
    }
}

// only square image that are power of 2 are supported
std::vector<std::vector<std::complex<double>>> kernel(bool direct, const std::vector<std::vector<std::complex<double>>> &input)
{
    // Dimensions of the 2D input
    int rows = input.size();
    int cols = input[0].size();

    // Determine grid size
    int grid_size = cols / THREAD_PER_BLOCK;
    if (cols % THREAD_PER_BLOCK != 0)
        grid_size++;

    std::cout << "\nGPU_PARALLEL\n";
    auto start_parallel = std::chrono::high_resolution_clock::now();

    int log_cols = (int)(log(cols) / log(2));
    int log_rows = (int)(log(rows) / log(2));

    // Device pointers for 2D arrays (stored as 1D linear memory)
    cuDoubleComplex *d_a; // Input matrix
    cuDoubleComplex *d_y; // Output matrix
    cuDoubleComplex *d_x; // Temporary matrix for intermediate results
    cuDoubleComplex *d_t; // Temporary matrix for intermediate results
    int *d_atomic_array;
    int *d_first_computation_j;

    // Allocate device memory
    cudaMallocManaged((void **)&d_a, sizeof(cuDoubleComplex) * rows * cols);
    cudaMallocManaged((void **)&d_y, sizeof(cuDoubleComplex) * rows * cols);
    cudaMallocManaged((void **)&d_x, sizeof(cuDoubleComplex) * rows * cols);
    cudaMallocManaged((void **)&d_t, sizeof(cuDoubleComplex) * rows * cols);
    cudaMallocManaged((void **)&d_atomic_array, sizeof(int) * (log_cols + 1));
    cudaMallocManaged((void **)&d_first_computation_j, sizeof(int) * rows);

    // Copy 2D input to device as 1D array
    // TODO :
    // if the input is not square, the remaining elements will be zero
    std::vector<cuDoubleComplex> h_a(rows * cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            d_a[i * cols + j] = make_cuDoubleComplex(input[i][j].real(), input[i][j].imag());
        }
    }

    auto malloc_complete = std::chrono::high_resolution_clock::now();
    bool tmp_direct = true;
    // Step 1: Perform row-wise FFT
    compute_row_wise_fft(tmp_direct, grid_size, cols, log_cols, d_a, d_y, d_x, d_t, d_atomic_array, d_first_computation_j);
    // cudaDeviceSynchronize();

    // Step 2: perform the tranpose of d_y in d_a
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            d_a[j * rows + i] = d_y[i * cols + j];
        }
    }

    // Step 3: Perform column-wise FFT
    compute_row_wise_fft(tmp_direct, grid_size, rows, log_rows, d_a, d_y, d_x, d_t, d_atomic_array, d_first_computation_j);
    // cudaDeviceSynchronize();

    // Step 4: tranpose the result of the column wise fft in d_a
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            d_a[j * rows + i] = d_y[i * cols + j];
        }
    }
    //  output is in d_a

    auto end_parallel = std::chrono::high_resolution_clock::now();

    if (direct)
    {
        std::chrono::duration<double> duration_parallel = end_parallel - start_parallel;
        std::chrono::duration<double> duration_parallel_without_malloc = end_parallel - malloc_complete;
        std::cout << "Parallel 2D FFT execution time: " << duration_parallel.count() << " seconds" << std::endl;
        std::cout << "Parallel 2D FFT execution time WITHOUT MALLOC: " << duration_parallel_without_malloc.count() << " seconds" << std::endl;

        // Convert to 2D vector of std::complex
        std::vector<std::vector<std::complex<double>>> output(rows, std::vector<std::complex<double>>(cols));
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                output[i][j] = std::complex<double>(cuCreal(d_a[i * cols + j]), cuCimag(d_a[i * cols + j]));
            }
        }
        // Free device memory
        cudaFree(d_a);
        cudaFree(d_y);
        cudaFree(d_x);
        cudaFree(d_t);
        cudaFree(d_atomic_array);
        cudaFree(d_first_computation_j);
        return output;
    }

    tmp_direct = false;
    // Step 1: Perform row-wise FFT
    compute_row_wise_fft(tmp_direct, grid_size, cols, log_cols, d_a, d_y, d_x, d_t, d_atomic_array, d_first_computation_j);

    // Step 2: perform the tranpose of d_y in d_a
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            d_a[j * rows + i] = d_y[i * cols + j];
        }
    }

    // Step 3: Perform column-wise FFT
    compute_row_wise_fft(tmp_direct, grid_size, rows, log_rows, d_a, d_y, d_x, d_t, d_atomic_array, d_first_computation_j);

    // Step 4: tranpose the result of the column wise fft in d_a
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            d_a[j * rows + i] = d_y[i * cols + j];
        }
    }
    //  output is in d_a
    end_parallel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_parallel = end_parallel - start_parallel;
    std::chrono::duration<double> duration_parallel_without_malloc = end_parallel - malloc_complete;
    std::cout << "Parallel 2D FFT execution time: " << duration_parallel.count() << " seconds" << std::endl;
    std::cout << "Parallel 2D FFT execution time WITHOUT MALLOC: " << duration_parallel_without_malloc.count() << " seconds" << std::endl;

    // Convert to 2D vector of std::complex
    std::vector<std::vector<std::complex<double>>> output(rows, std::vector<std::complex<double>>(cols));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            output[i][j] = std::complex<double>(cuCreal(d_a[i * cols + j]) / (rows * cols), cuCimag(d_a[i * cols + j]) / (rows * cols));
        }
    }
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_y);
    cudaFree(d_x);
    cudaFree(d_t);
    cudaFree(d_atomic_array);
    cudaFree(d_first_computation_j);
    return output;
}
