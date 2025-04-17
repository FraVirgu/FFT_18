#ifndef FFT2D_HPP
#define FFT2D_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>

class FFT2D {
public:

    static std::vector<std::vector<std::complex<double>>> forward(const cv::Mat& input);
    static cv::Mat inverse(const std::vector<std::vector<std::complex<double>>>& input);
};

#endif
