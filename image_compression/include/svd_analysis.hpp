#ifndef SVD_ANALYSIS_HPP
#define SVD_ANALYSIS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class SVDAnalyzer {
public:
    SVDAnalyzer(const std::string& imagePath, int n);

    void computeSVD();
    std::vector<float> getSingularValues() const;
    void saveSingularValues(const std::string& filename) const;
    void showOriginalImage() const;

private:
    std::string imagePath;
    int size;
    cv::Mat originalGray;
    cv::Mat U, S, VT;
    std::vector<float> singularValues;
};

#endif
