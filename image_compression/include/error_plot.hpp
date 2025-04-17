#ifndef ERROR_PLOT_HPP
#define ERROR_PLOT_HPP

#include <vector>
#include <complex>
#include <string>
#include <opencv2/opencv.hpp>

struct ErrorData {
    double error;
    int frequenciesEliminated;
};

class ErrorPlot {
public:
    ErrorPlot(const cv::Mat& originalImage);

    void computeErrorsForThresholds(const std::vector<double>& thresholds);
    void saveToCSV(const std::string& filename) const;

private:
    int n;
    cv::Mat original;
    std::vector<std::pair<double, ErrorData>> errorResults;
};

#endif
