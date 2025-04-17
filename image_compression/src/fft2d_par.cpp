#include "../include/fft2d_par.hpp"
#include "../../FFT/include/Cooley-Tukey-parallel.hpp"
#include <mpi.h>

std::vector<std::vector<std::complex<double>>> FFT2DParallel::forward(const cv::Mat& input) {
    int rows = input.rows;
    int cols = input.cols;
    ParallelIterativeFFT fft;

    std::vector<std::vector<std::complex<double>>> temp(rows, std::vector<std::complex<double>>(cols));
    std::vector<std::vector<std::complex<double>>> result(rows, std::vector<std::complex<double>>(cols));

    // FFT on rows
    for (int i = 0; i < rows; ++i) {
        std::vector<std::complex<double>> row(cols);
        for (int j = 0; j < cols; ++j)
            row[j] = std::complex<double>(input.at<float>(i, j), 0);
        temp[i] = fft.findFFT(row);
    }

    // FFT on columns
    for (int j = 0; j < cols; ++j) {
        std::vector<std::complex<double>> col(rows);
        for (int i = 0; i < rows; ++i)
            col[i] = temp[i][j];
        std::vector<std::complex<double>> colFFT = fft.findFFT(col);
        for (int i = 0; i < rows; ++i)
            result[i][j] = colFFT[i];
    }

    return result;
}

cv::Mat FFT2DParallel::inverse(const std::vector<std::vector<std::complex<double>>>& input) {
    int rows = input.size();
    int cols = input[0].size();
    SequentialFFT fft;

    std::vector<std::vector<std::complex<double>>> temp(rows, std::vector<std::complex<double>>(cols));
    std::vector<std::vector<std::complex<double>>> result(rows, std::vector<std::complex<double>>(cols));

    // IFFT on columns
    for (int j = 0; j < cols; ++j) {
        std::vector<std::complex<double>> col(rows);
        for (int i = 0; i < rows; ++i)
            col[i] = input[i][j];
        std::vector<std::complex<double>> colIFFT = fft.iterative_inverse_FFT(col);
        for (int i = 0; i < rows; ++i)
            temp[i][j] = colIFFT[i];
    }

    // IFFT on rows
    for (int i = 0; i < rows; ++i)
        result[i] = fft.iterative_inverse_FFT(temp[i]);

    cv::Mat output(rows, cols, CV_32F);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            output.at<float>((i + 1) % rows, (j + 1) % cols) = std::real(result[i][j]);

    cv::flip(output, output, -1);
    return output;
}