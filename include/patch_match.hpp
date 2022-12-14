#pragma once
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "tictoc.hpp"

#define TERMINATION 1
#define TSS_STEP 0.75

enum SearchOption: int {
    FULL = 0,
    L2R = 1,
    R2L = 2
};

extern std::unique_ptr<TicToc> timer;

const std::array<cv::Point2f, 16> DIRECTS = {
    cv::Point2f(-1, 1), cv::Point2f(-0.5, 1), cv::Point2f(0, 1), cv::Point2f(0.5, 1), cv::Point2f(1, 1),
    cv::Point2f(-1, 0.5),                                                             cv::Point2f(1, 0.5), 
    cv::Point2f(-1, 0),                                                               cv::Point2f(1, 0), 
    cv::Point2f(-1, -0.5),                                                            cv::Point2f(1, -0.5), 
    cv::Point2f(-1, -1), cv::Point2f(-0.5, -1), cv::Point2f(0, -1), cv::Point2f(0.5, -1), cv::Point2f(1, -1)        // 中心在最后
};

inline void border_padding_size(int &rows, int &cols, int patch_size) {
    int row_r = rows % patch_size, col_r = cols % patch_size;
    if (row_r > 0) {
        rows += row_r;
    }
    if (col_r > 0) {
        cols += col_r;
    }
}

// 金字塔搜索（上一次搜索结果可以作为本次搜索结果的初始化）
cv::Mat pyramid_searching(
    const cv::Mat& prev, const cv::Mat& next, cv::Mat& arrow, cv::Mat& output, 
    int patch_radius, int pyramid_lv = 4, bool only_col = false, bool blur = false, SearchOption l2r = SearchOption::FULL
);

// N步搜索法，可以配置为3步
void multi_step_searching(const cv::Mat& prev, const cv::Mat& next, cv::Mat& arrow, cv::Mat& output, int step_size, int patch_radius = 2);

// 穷举法
void exhaustive_search(const cv::Mat& prev, const cv::Mat& next, cv::Mat& arrow, cv::Mat& output, int patch_radius);