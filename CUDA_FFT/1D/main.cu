#include "./header.hpp"

std::vector<std::complex<double>> generate_sinusoidal_1d(int dim, double frequency, double amplitude = 1.0)
{
    std::vector<std::complex<double>> input;

    for (int i = 0; i < dim; i++)
    {
        // Sinusoide: amplitude * sin(2 * pi * f * t)
        double value = amplitude * sin(2.0 * M_PI * frequency * i / dim);
        input.push_back(std::complex<double>(value, 0.0)); // Solo parte reale
    }

    return input;
}

int main()
{
    int *prova;
    cudaMallocManaged((void **)&prova, sizeof(int));
    srand(95);

    std::vector<std::complex<double>> input;

    /*
       for (int i = 0; i < N; i++)
    {
        // Generate random numbers between -1.0 and 1.0
        double real_part = (rand() % (RAND_MAX)) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
        double imag_part = (rand() % (RAND_MAX)) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
        input.push_back(std::complex<double>(real_part, imag_part));
    }

    */

    input = generate_sinusoidal_1d(N, 0.5);

    // DIRECT TRANSFORM
    bool direct = true;
    std::vector<std::complex<double>> cuda_output_vector = kernel(direct, input);
    plot_fft_result(cuda_output_vector);
    /*
    // DIRECT + INVERSE TRANSFORM
     bool direct = false;
     std::vector<std::complex<double>> inverse_cuda_output_vector = kernel(direct, input);
     // compareComplexVectors(inverse_cuda_output_vector, input);


    */

    return 0;
}
