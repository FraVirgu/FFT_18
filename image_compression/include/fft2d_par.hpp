// fft2d_parallel.hpp
#ifndef FFT2D_PARALLEL_HPP
#define FFT2D_PARALLEL_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>

class FFT2DParallel {
public:

    static std::vector<std::vector<std::complex<double>>> forward(const cv::Mat& input);

    static cv::Mat inverse(const std::vector<std::vector<std::complex<double>>>& input);
};

#endif // FFT2D_PARALLEL_HPP
