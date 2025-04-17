#include "error_plot.hpp"
#include "fft2d.hpp"
#include <fstream>
#include <cmath>
#include <iostream>

ErrorPlot::ErrorPlot(const cv::Mat& originalImage) {
    originalImage.convertTo(original, CV_32F);
    n = original.rows;
}

void ErrorPlot::computeErrorsForThresholds(const std::vector<double>& thresholds) {
    errorResults.clear();
    auto fftFull = FFT2D::forward(original);

    for (double t : thresholds) {
        auto filtered = fftFull;
        int zeroed = 0;

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                if (std::abs(filtered[i][j]) < t) {
                    filtered[i][j] = 0;
                    ++zeroed;
                }
            }

        cv::Mat reconstructed = FFT2D::inverse(filtered);

        double error = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                error += std::pow(original.at<float>(i, j) - reconstructed.at<float>(i, j), 2);

        error = std::sqrt(error) / (n * n);
        errorResults.emplace_back(t, ErrorData{error, zeroed});
    }
}

void ErrorPlot::saveToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << filename << std::endl;
        return;
    }

    file << "Threshold,Error,FrequenciesEliminated\n";
    for (const auto& pair : errorResults)
        file << pair.first << "," << pair.second.error << "," << pair.second.frequenciesEliminated << "\n";

    file.close();
}
