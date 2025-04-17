#include "fft_analysis.hpp"
#include "svd_analysis.hpp"
#include "error_plot.hpp"
#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    const int size = 256;
    double threshold = 1000.0;


    const std::string imagePath = argv[1];

    // SVD ANALYSIS
    SVDAnalyzer svd(imagePath, size);
    svd.computeSVD();
    svd.saveSingularValues("singular_values.csv");
    svd.showOriginalImage();

    // FFT ANALYSIS
    FFTAnalysis fft(size);
    fft.loadImage(imagePath);
    fft.computeFFT();
    fft.saveFFTToCSV("fft_output_2d.csv");
    fft.saveMagnitudeToCSV("fft_magnitude.csv");
    fft.showMagnitudeSpectrum(false);

    fft.applyThreshold(threshold);
    fft.saveMagnitudeToCSV("fft_magnitude_filtered.csv", true);
    fft.showMagnitudeSpectrum(true);

    fft.computeIFFT();
    fft.showReconstructedImage();
    fft.computeReconstructionError();

    std::cout << "Reconstruction error for threshold " << threshold << ": "
              << fft.getError() << std::endl;

    // ERROR PLOT VS THRESHOLD
    cv::Mat input = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    cv::resize(input, input, cv::Size(size, size));
    ErrorPlot errorPlot(input);
    errorPlot.computeErrorsForThresholds({10, 50, 100, 200, 500, 1000});
    errorPlot.saveToCSV("error_vs_threshold.csv");

    std::cout << "Everything Completed and data saved\n";

    return 0;
}
