#include "./header.hpp"
#include "main_class_2d.hpp"
#include "sinusoidal_2d.hpp"
#include "gaussian_2d.hpp"
#include "random_2d.hpp"
#include <vector>
#include <complex>
#include <cmath>

int MainClass2D::rows = 0;
int MainClass2D::cols = 0;

int main()
{
    int *prova;
    cudaMallocManaged((void **)&prova, sizeof(int));
    srand(95);

    MainClass2D::initializeParameters(1024, 1024);

    Sinusoidal2D sinusoidalGen;
    Gaussian2D gaussianGen;
    Random2D randomGen;

    // Example usage
    auto input_sinusoidal = sinusoidalGen.createInput();
    auto input_gaussian = gaussianGen.createInput();
    auto input_random = randomGen.createInput();

    // DIRECT TRANSFORM
    bool direct = true;
    std::vector<std::vector<std::complex<double>>> cuda_output_vector = direct_fft_2d(input_sinusoidal);
    save_fft_result_2d(cuda_output_vector);

    // INVERSE TRANSFORM
    std::vector<std::vector<std::complex<double>>> cuda_output_vector_inverse = inverse_fft_2d(cuda_output_vector);

    return 0;
}