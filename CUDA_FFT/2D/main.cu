#include "./header.hpp"
#include <vector>
#include <complex>
#include <cmath>

// Funzione per generare un'onda sinusoidale 2D
std::vector<std::vector<std::complex<double>>> generate_sinusoidal_2d(int rows, int cols, double frequency_x, double frequency_y)
{
    std::vector<std::vector<std::complex<double>>> input(rows, std::vector<std::complex<double>>(cols));

    // Genera l'onda sinusoidale 2D
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Sinusoide 2D: sin(2 * pi * f_x * x) * sin(2 * pi * f_y * y)
            double value = sin(2.0 * M_PI * frequency_x * i / rows) * sin(2.0 * M_PI * frequency_y * j / cols);
            input[i][j] = std::complex<double>(value, 0.0); // Solo parte reale
        }
    }

    return input;
}

std::vector<std::vector<std::complex<double>>> generate_gaussian_2d(int rows, int cols, double sigma_x, double sigma_y)
{
    std::vector<std::vector<std::complex<double>>> input(rows, std::vector<std::complex<double>>(cols));

    double center_x = rows / 2.0;
    double center_y = cols / 2.0;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Gaussiana 2D: e^(-(x^2 / (2 * sigma_x^2) + y^2 / (2 * sigma_y^2)))
            double x_term = pow(i - center_x, 2) / (2 * sigma_x * sigma_x);
            double y_term = pow(j - center_y, 2) / (2 * sigma_y * sigma_y);
            double value = exp(-(x_term + y_term));
            input[i][j] = std::complex<double>(value, 0.0);
        }
    }

    return input;
}

int main()
{
    int *prova;
    cudaMallocManaged((void **)&prova, sizeof(int));
    // Definizione delle dimensioni della matrice
    int rows = N;             // Numero di righe
    int cols = N;             // Numero di colonne
    double frequency_x = 5.0; // Frequenza lungo l'asse X
    double frequency_y = 5.0; // Frequenza lungo l'asse Y
    srand(95);

    /*
     // Creazione di una matrice 2D di input con numeri casuali
        std::vector<std::vector<std::complex<double>>> input(rows, std::vector<std::complex<double>>(cols));
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Generazione di numeri casuali tra -1.0 e 1.0
                double real_part = (rand() % RAND_MAX) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
                double imag_part = (rand() % RAND_MAX) / static_cast<double>(RAND_MAX) * 2.0 - 1.0;
                input[i][j] = std::complex<double>(real_part, imag_part);
            }
        }

    */

    // Genera l'onda sinusoidale 2D
    auto input = generate_sinusoidal_2d(rows, cols, frequency_x, frequency_y);
    input = generate_gaussian_2d(rows, cols, 100.0, 100.0);

    // Calcolo della FFT con la funzione kernel
    // direct == true => FFT
    // direct == false => FFT + IFFT
    bool direct = false;

    std::vector<std::vector<std::complex<double>>> cuda_output_vector = kernel(direct, input);
    save_fft_result_2d(cuda_output_vector);
    // std::vector<std::vector<std::complex<double>>> iterative_output_vector = iterative_FFT_2D(input);
    // compareComplexMatrices(input, cuda_output_vector);

    // cuda_library_fft_2d(input);

    return 0;
}
