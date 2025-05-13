#include "../include/svd_analysis.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>

SVDAnalyzer::SVDAnalyzer(const std::string& imagePath, int n)
    : imagePath(imagePath), size(n) {

    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Impossible to upload image: " << imagePath << std::endl;
        exit(1);
    }

    cv::resize(img, originalGray, cv::Size(n, n));
    originalGray.convertTo(originalGray, CV_32F);
}

void SVDAnalyzer::computeSVD() {
    cv::SVD::compute(originalGray, S, U, VT);
    singularValues.clear();
    for (int i = 0; i < S.rows; i++) {
        singularValues.push_back(S.at<float>(i));
    }
}

std::vector<float> SVDAnalyzer::getSingularValues() const {
    return singularValues;
}

void SVDAnalyzer::saveSingularValues(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Impossible to open file: " << filename << std::endl;
        return;
    }

    for (float val : singularValues) {
        file << val << std::endl;
    }

    file.close();
}

void SVDAnalyzer::showOriginalImage() const {
    cv::Mat normImg;
    cv::normalize(originalGray, normImg, 0, 255, cv::NORM_MINMAX);
    normImg.convertTo(normImg, CV_8U);
    cv::imwrite("../output/image_output/original_image.png", normImg);
}
