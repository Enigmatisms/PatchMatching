#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "include/patch_match.hpp"

void multi_step_main(int argc, char** argv) {
    int max_range = 32;
    int max_radius = 4, pad_size = max_range + max_radius;
    cv::Mat prev_img = cv::imread("../prev.bmp");
    cv::Mat next_img = cv::imread("../next.bmp");
    cv::resize(prev_img, prev_img, prev_img.size() * 2);
    cv::resize(next_img, next_img, next_img.size() * 2);
    cv::Mat out = prev_img.clone();
    cv::Mat arrow(prev_img.rows, prev_img.cols, CV_8UC3);
    cv::copyMakeBorder(prev_img, prev_img, pad_size, pad_size, pad_size, pad_size, cv::BORDER_REFLECT);
    cv::copyMakeBorder(next_img, next_img, pad_size, pad_size, pad_size, pad_size, cv::BORDER_REFLECT);
    
    TicToc timer;
    multi_step_searching(prev_img, next_img, arrow, out, max_range, max_radius, &timer);

    cv::imshow("optic-flow", out);
    cv::imshow("arrow", arrow);
    cv::waitKey(0);
}

void exhaustive_main(int argc, char** argv) {
    int patch_size = 16;
    cv::Mat prev_img = cv::imread("../prev.bmp");
    cv::Mat next_img = cv::imread("../next.bmp");
    cv::resize(prev_img, prev_img, prev_img.size() * 2);
    cv::resize(next_img, next_img, next_img.size() * 2);
    int rows = prev_img.rows, cols = prev_img.cols;
    border_padding_size(rows, cols, patch_size);
    rows -= prev_img.rows;
    cols -= prev_img.cols;
    
    cv::copyMakeBorder(prev_img, prev_img, 0, rows, 0, cols, cv::BORDER_REFLECT);
    cv::copyMakeBorder(next_img, next_img, 0, rows, 0, cols, cv::BORDER_REFLECT);
    cv::Mat out = prev_img.clone();
    cv::Mat arrow = prev_img.clone();

    TicToc timer;
    exhaustive_search(prev_img, next_img, arrow, out, patch_size, &timer);

    cv::imshow("optic-flow", out);
    cv::imshow("arrow", arrow);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    multi_step_main(argc, argv);
    return 0;
}