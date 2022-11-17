#include "../include/patch_match.hpp"

std::unique_ptr<TicToc> timer;

// Cost is based on SAD
template<typename MatType>
float get_image_cost(const cv::Mat& target, MatType&& inquiry) {
    cv::Mat output(target.size(), CV_32FC3);
    // cv::normalize(inquiry, norm_inq);
    // cv::normalize(target, norm_tar);
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
    return recursive_search(prev_target, next, min_mv, cost, step_size * TSS_STEP, patch_radius, rows, cols);
}

void multi_step_searching(const cv::Mat& prev, const cv::Mat& next, cv::Mat& arrow, cv::Mat& output, int step_size, int patch_radius) {
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
void exhaustive_search(const cv::Mat& prev, const cv::Mat& next, cv::Mat& arrow, cv::Mat& output, int patch_size) {
    int rows = prev.rows, cols = prev.cols;
    int row_ps = rows / patch_size, col_ps = cols / patch_size;
    if (timer != nullptr) {
        timer->tic();
    }
    #pragma omp parallel for num_threads(8)
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

cv::Mat pyramid_searching(
    const cv::Mat& prev, const cv::Mat& next, cv::Mat& arrow, cv::Mat& output,
    int patch_radius, int pyramid_lv, bool only_row, bool blur, SearchOption l2r
) {
    cv::Size start_size = prev.size();
    cv::Mat img_offset;
    bool pyramid_ready = false;
    if (timer != nullptr) {
        timer->tic();
    }
    int padding_size = patch_radius * 2;
    for (int lv = 0; lv < pyramid_lv; lv++) {
        int left_mv = (pyramid_lv - lv - 1);
        cv::Size new_size = cv::Size(start_size.width >> left_mv, start_size.height >> left_mv);
        if (pyramid_ready == false) {
            img_offset.create(new_size, CV_16SC2);
        } else {
            // 已经保证了输入图像原始大小可以被8整除（padding）
            if (blur == true) {
                cv::blur(img_offset, img_offset, cv::Size(3, 3));
            }
            cv::resize(img_offset, img_offset, new_size, 0., 0., cv::INTER_LINEAR);
            img_offset *= 2.0;
        }
        cv::Mat p_prev, p_next;
        cv::resize(prev, p_prev, new_size);
        cv::resize(next, p_next, new_size);
        cv::copyMakeBorder(p_prev, p_prev, padding_size, padding_size, padding_size, padding_size, cv::BORDER_REFLECT);
        cv::copyMakeBorder(p_next, p_next, padding_size, padding_size, padding_size, padding_size, cv::BORDER_REFLECT);
        cv::Point offset(padding_size, padding_size);
        int max_cols = p_prev.cols - padding_size, max_rows = p_prev.rows - padding_size;

        printf("Max row: %d, max col: %d, padding size: %d\n", max_rows, max_cols, padding_size);
        int row_search = (only_row == true) ? 0 : patch_radius;
        #pragma omp parallel for num_threads(8)
        for (int row = padding_size; row < max_rows; row++) {
            for (int col = padding_size; col < max_cols; col++) {
                cv::Point p_anchor(col, row), actual_index = p_anchor - offset;
                cv::Mat p_pat = center_extract(p_prev, p_anchor, patch_radius);
                float min_cost = 1e9;
                cv::Vec2s min_mv(0, 0);
                int guide_x = 0, guide_y = 0;
                if (pyramid_ready == true) {
                    cv::Vec2s tmp = img_offset.at<cv::Vec2s>(actual_index);
                    guide_x = tmp[0];
                    guide_y = tmp[1];
                }
                int col_start = (l2r < 2) ? -patch_radius : 0;
                int col_end = col_start + ((l2r == 0) ? (patch_radius << 1) : patch_radius);
                for (int i = -row_search; i <= row_search; i++) {
                    for (int j = col_start; j <= col_end; j++) {
                        cv::Mat n_pat = center_extract(p_next, p_anchor + cv::Point(j + guide_x, i + guide_y), patch_radius);
                        float cost = get_image_cost(p_pat, n_pat);
                        if (cost < min_cost) {
                            min_cost = cost;
                            min_mv[0] = j + guide_x;
                            min_mv[1] = i + guide_y;
                        }
                    }
                }
                img_offset.at<cv::Vec2s>(actual_index) = min_mv;
            }
        }
        padding_size += (patch_radius << (lv + 1));
        pyramid_ready = true;
    }
    if (timer != nullptr) {
        printf("Running time: %.4lf ms\n", timer->toc());
    }
    output.forEach<cv::Vec3b>(
        [&prev, &img_offset, &start_size](cv::Vec3b& color, const int* pos) -> void {
            cv::Vec2s point_mv = img_offset.at<cv::Vec2s>(pos[0], pos[1]);      // TODO
            int x = pos[1] - point_mv[0], y = pos[0] - point_mv[1];
            if (x >= 0 && x < start_size.width && y >= 0 && y < start_size.height) {
                color = prev.at<cv::Vec3b>(y, x);
            } else {
                color = prev.at<cv::Vec3b>(pos[0], pos[1]);
            }
        }
    );
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < start_size.height; i += 20) {
        for (int j = 0; j < start_size.width; j += 20) {
            cv::Vec2s point_mv = img_offset.at<cv::Vec2s>(i, j);
            cv::Point sp(j, i), ep(j + point_mv[0], i + point_mv[1]);
            cv::arrowedLine(arrow, sp, ep, cv::Scalar(0, 255, 0));
        }
    }
    printf("Process completed.\n");
    return img_offset;
}
