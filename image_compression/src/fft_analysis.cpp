#include "../include/fft_analysis.hpp"
#include "fft2d.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <fstream>

FFTAnalysis::FFTAnalysis(int size) : n(size), reconstructionError(0.0) {}

void FFTAnalysis::loadImage(const std::string& path) {
    originalImage = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (originalImage.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return;
    }
    cv::resize(originalImage, originalImage, cv::Size(n, n));
    originalImage.convertTo(originalFloat, CV_32F);
}

void FFTAnalysis::computeSVD() {
    cv::Mat w, u, vt;
    cv::SVD::compute(originalFloat, w, u, vt);
    singularValues.assign((float*)w.datastart, (float*)w.dataend);
}

void FFTAnalysis::computeFFT() {
    fft2D = FFT2D::forward(originalFloat);
    filteredFFT2D = fft2D;
}

void FFTAnalysis::applyThreshold(double threshold) {
    filteredFFT2D = fft2D;
    int zeroed = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (std::abs(filteredFFT2D[i][j]) < threshold) {
                filteredFFT2D[i][j] = 0;
                zeroed++;
            }
        }
    }
    std::cout << "Frequencies zeroed below threshold " << threshold << ": " << zeroed << " out of " << (n * n) << std::endl;
}

void FFTAnalysis::computeIFFT() {
    reconstructedImage = FFT2D::inverse(filteredFFT2D);
}

void FFTAnalysis::computeReconstructionError() {
    double error = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            error += std::pow(originalFloat.at<float>(i, j) - reconstructedImage.at<float>(i, j), 2);

    reconstructionError = std::sqrt(error) / (n * n);
}

void FFTAnalysis::showOriginalImage() const {
    cv::imwrite("original_image.png", originalImage);
}

void FFTAnalysis::showMagnitudeSpectrum(bool afterFilter) const {
    const auto& data = afterFilter ? filteredFFT2D : fft2D;
    cv::Mat mag(n, n, CV_32F);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mag.at<float>(i, j) = std::log(1 + std::abs(data[i][j]));

    double minVal, maxVal;
    cv::minMaxLoc(mag, &minVal, &maxVal);
    cv::Mat magNorm;
    mag.convertTo(magNorm, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    std::string filename = afterFilter ? "filtered_fft_magnitude.png" : "fft_magnitude.png";
    cv::imwrite(filename, magNorm);

    if (afterFilter) {
        cv::Mat diff(n, n, CV_32F);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                diff.at<float>(i, j) = std::log(1 + std::abs(std::abs(fft2D[i][j]) - std::abs(filteredFFT2D[i][j])));

        cv::minMaxLoc(diff, &minVal, &maxVal);
        cv::Mat diffNorm;
        diff.convertTo(diffNorm, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        cv::imwrite("fft_removed_frequencies.png", diffNorm);
    }
}

void FFTAnalysis::showReconstructedImage() const {
    cv::Mat normImg;
    cv::normalize(reconstructedImage, normImg, 0, 255, cv::NORM_MINMAX);
    normImg.convertTo(normImg, CV_8U);
    cv::imwrite("reconstructed_image.png", normImg);
}

double FFTAnalysis::getError() const {
    return reconstructionError;
}

const std::vector<std::vector<std::complex<double>>>& FFTAnalysis::getFFTData() const {
    return fft2D;
}

void FFTAnalysis::saveFFTToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Impossible to open file: " << filename << std::endl;
        return;
    }
    for (const auto& row : fft2D) {
        for (size_t j = 0; j < row.size(); j++) {
            file << "(" << std::real(row[j]) << "," << std::imag(row[j]) << ")";
            if (j != row.size() - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "FFT 2D saved in " << filename << std::endl;
}

void FFTAnalysis::saveMagnitudeToCSV(const std::string& filename, bool afterFilter) const {
    const auto& data = afterFilter ? filteredFFT2D : fft2D;

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Impossible to open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double magnitude = std::abs(data[i][j]);
            file << magnitude;
            if (j != n - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "FFT Module saved in " << filename << std::endl;
}


