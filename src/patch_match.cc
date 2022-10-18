#include "../include/patch_match.hpp"

float get_image_cost(const cv::Mat& target, const cv::Mat& inquiry) {
    cv::Mat output(target.size(), CV_32FC3);
    cv::absdiff(target, inquiry, output);
    cv::Scalar result = cv::sum(output);
    return 0.333 * (result[0] + result[1] + result[2]);
}

float get_image_cost(const cv::Mat& target, cv::Mat&& inquiry) {
    cv::Mat output(target.size(), CV_32FC3);
    cv::absdiff(target, inquiry, output);
    cv::Scalar result = cv::sum(output);
    return 0.333 * (result[0] + result[1] + result[2]);
}

cv::Mat center_extract(const cv::Mat& img, const cv::Point& center, int radius) {
    cv::Point p_start = center - cv::Point(radius, radius);
    cv::Point p_end = center + cv::Point(radius + 1, radius + 1);
    cv::Mat roi = img(cv::Rect(p_start, p_end)), result;
    roi.convertTo(result, CV_32FC3);
    return result;
}

cv::Mat corner_extract(const cv::Mat& img, const cv::Point& center, int size) {
    cv::Point p_start = center;
    cv::Point p_end = center + cv::Point(size, size);
    cv::Mat roi = img(cv::Rect(p_start, p_end)), result;
    roi.convertTo(result, CV_32FC3);
    return result;
}

cv::Point recursive_search(const cv::Mat& prev_target, const cv::Mat& next, cv::Point center, float cost, int step_size, int patch_radius, int rows, int cols) {
    if (step_size <= TERMINATION) {
        return center;
    }
    cv::Point min_mv = center;
    for (int k = 0; k < 16; k++) {
        cv::Point2f float_mv = DIRECTS[k] * step_size;
        cv::Point mv = cv::Point(float_mv.x, float_mv.y) + center;
        if (mv.x >= patch_radius && mv.x < cols - patch_radius && mv.y >= patch_radius && mv.y < rows - patch_radius) {
            cv::Mat n_pat = center_extract(next, mv, patch_radius);
            float local_cost = get_image_cost(prev_target, n_pat);
            if (local_cost < cost) {
                cost = local_cost;
                min_mv = mv;
            }
        }
    }
    return recursive_search(prev_target, next, min_mv, cost, step_size * 0.75, patch_radius, rows, cols);
}

void multi_step_searching(const cv::Mat& prev, const cv::Mat& next, cv::Mat& arrow, cv::Mat& output, int step_size, int patch_radius, TicToc* timer) {
    int rows = prev.rows, cols = prev.cols;
    int start_offset = step_size + patch_radius;
    int max_rows = rows - start_offset, max_cols = cols - start_offset;
    cv::Point offset(start_offset, start_offset);
    cv::Rect roi(offset, cv::Point(max_cols, max_rows));
    arrow = prev.clone()(roi);
    if (timer != nullptr) {
        timer->tic();
    }
    #pragma omp parallel for num_threads(16)
    for (int i = start_offset; i < max_rows; i++) {
        for (int j = start_offset; j < max_cols; j++) {
            cv::Point prev_anchor(j, i);
            cv::Mat p_pat = center_extract(prev, prev_anchor, patch_radius);
            float center_cost = get_image_cost(p_pat, center_extract(next, prev_anchor, patch_radius));
            cv::Point best_estimate = recursive_search(p_pat, next, prev_anchor, center_cost, step_size, patch_radius, rows, cols);
            cv::Point diff = best_estimate - prev_anchor;
            prev_anchor -= offset;
            cv::Point t_pos = prev_anchor + diff;
            if (timer == nullptr) {
                if (t_pos.x >= 0 && t_pos.y >= 0 && t_pos.x < output.cols && t_pos.y < output.rows) {
                    output.at<cv::Vec3b>(t_pos) = prev.at<cv::Vec3b>(prev_anchor + offset);
                }
                if (i % 20 == 0 && j % 20 == 0) {
                    cv::arrowedLine(arrow, prev_anchor, prev_anchor + diff, cv::Scalar(0, 255, 0));
                }
            }
        }
    }
    if (timer != nullptr) {
        printf("Running time: %.4lf ms\n", timer->toc());
    }
    printf("Process completed.\n");
}

// 穷举法无需padding
void exhaustive_search(const cv::Mat& prev, const cv::Mat& next, cv::Mat& arrow, cv::Mat& output, int patch_size, TicToc* timer) {
    int rows = prev.rows, cols = prev.cols;
    int row_ps = rows / patch_size, col_ps = cols / patch_size;
    if (timer != nullptr) {
        timer->tic();
    }
    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < row_ps; i++) {
        for (int j = 0; j < col_ps; j++) {
            cv::Point current_pos = cv::Point(j, i) * patch_size;
            cv::Mat p_pat = corner_extract(prev, current_pos, patch_size);
            cv::Point target_pos(0, 0);
            float min_cost = 1e9;
            for (int m = 0; m < row_ps; m++) {
                for (int n = 0; n < col_ps; n++) {
                    cv::Point pos = cv::Point(n, m) * patch_size;
                    cv::Mat n_pat = corner_extract(next, pos, patch_size);
                    float cost = get_image_cost(p_pat, n_pat);
                    if (cost < min_cost) {
                        min_cost = cost;
                        target_pos = pos;
                    }
                }
            }
            if (timer == nullptr) {
                p_pat.copyTo(output.rowRange(target_pos.y, target_pos.y + patch_size).colRange(target_pos.x, target_pos.x + patch_size));
                cv::arrowedLine(arrow, current_pos, target_pos, cv::Scalar(0, 255, 0));
            }
        }
    }
    if (timer != nullptr) {
        printf("Running time: %.4lf ms\n", timer->toc());
    }
    printf("Process completed.\n");
}
