#ifndef FFT_ANALYSIS_HPP
#define FFT_ANALYSIS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <string>

class FFTAnalysis {
public:
    FFTAnalysis(int size);

    void loadImage(const std::string& path);
    void computeSVD();
    void computeFFT();
    void applyThreshold(double threshold);
    void computeIFFT();
    void computeReconstructionError();

    void showOriginalImage() const;
    void showMagnitudeSpectrum(bool afterFilter = false) const;
    void showReconstructedImage() const;

    double getError() const;
    const std::vector<std::vector<std::complex<double>>>& getFFTData() const;

    void saveFFTToCSV(const std::string& filename) const;
    void saveMagnitudeToCSV(const std::string& filename, bool afterFilter = false) const;



private:
    int n;
    cv::Mat originalImage;
    cv::Mat originalFloat;
    cv::Mat reconstructedImage;
    std::vector<std::vector<std::complex<double>>> fft2D;
    std::vector<std::vector<std::complex<double>>> filteredFFT2D;

    std::vector<float> singularValues;
    double reconstructionError;
};

#endif
