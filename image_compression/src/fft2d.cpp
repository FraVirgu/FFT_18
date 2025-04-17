#include "fft2d.hpp"
#include "Cooley-Tukey.hpp"
#include <iostream>

std::vector<std::vector<std::complex<double>>> FFT2D::forward(const cv::Mat& input) {
    int rows = input.rows;
    int cols = input.cols;

    SequentialFFT fft1D;
    std::vector<std::vector<std::complex<double>>> temp(rows, std::vector<std::complex<double>>(cols));
    std::vector<std::vector<std::complex<double>>> result(rows, std::vector<std::complex<double>>(cols));

    for (int i = 0; i < rows; i++) {
        std::vector<std::complex<double>> row(cols);
        for (int j = 0; j < cols; j++)
            row[j] = std::complex<double>(input.at<float>(i, j), 0);

        temp[i] = fft1D.iterative_FFT(row);
    }

    for (int j = 0; j < cols; j++) {
        std::vector<std::complex<double>> col(rows);
        for (int i = 0; i < rows; i++)
            col[i] = temp[i][j];

        std::vector<std::complex<double>> colFFT = fft1D.iterative_FFT(col);
        for (int i = 0; i < rows; i++)
            result[i][j] = colFFT[i];
    }

    return result;
}

cv::Mat FFT2D::inverse(const std::vector<std::vector<std::complex<double>>>& input) {
    int rows = input.size();
    int cols = input[0].size();

    SequentialFFT fft1D;
    std::vector<std::vector<std::complex<double>>> temp(rows, std::vector<std::complex<double>>(cols));
    std::vector<std::vector<std::complex<double>>> result(rows, std::vector<std::complex<double>>(cols));

    for (int j = 0 ; j < cols; j++) {
        std::vector<std::complex<double>> col(rows);
        for (int i = 0; i < rows; i++)
            col[i] = input[i][j];

        std::vector<std::complex<double>> colIFFT = fft1D.iterative_inverse_FFT(col);
        for (int i = 0; i < rows; i++)
            temp[i][j] = colIFFT[i];
    }

    for (int i = 0; i < rows; i++) {
        result[i] = fft1D.iterative_inverse_FFT(temp[i]);
    }

    cv::Mat output(rows, cols, CV_32F);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            output.at<float>(i, j) = std::real(result[i][j]);

    cv::flip(output, output, -1);

    return output;
}
