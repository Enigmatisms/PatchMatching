#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "include/patch_match.hpp"

inline float mse2psnr(float mse) {
    mse = fmax(1e-8, mse);
    return - 10.f * logf(mse) / logf(10.);
}

float image_psnr(const cv::Mat& src, const cv::Mat& dst) {
    cv::Mat src_f, dst_f;
    src.convertTo(src_f, CV_32FC3);
    dst.convertTo(dst_f, CV_32FC3);
    cv::Mat diff_mat = (src_f - dst_f) / 255.;
    cv::multiply(diff_mat, diff_mat, diff_mat); // power
    cv::Scalar mean = cv::mean(diff_mat);
    return (mse2psnr(mean[0]) + mse2psnr(mean[1]) + mse2psnr(mean[2])) / 3.;
}

void get_disparity_image(const cv::Mat& img_disp, cv::Mat& output) {
    cv::Mat channels[2];
    cv::split(img_disp, channels);
    cv::Mat disp_img = channels[0];
    disp_img.convertTo(disp_img, CV_32FC1);
    output.create(disp_img.size(), CV_8UC1);

    double min_val = 0.0, max_val = 0.0;
    cv::minMaxLoc(disp_img, &min_val, &max_val);
    printf("Minimum move: %lf, maximum move: %lf\n", min_val, max_val);
    disp_img = (disp_img - min_val) / (max_val - min_val);
    output.forEach<uchar>(
        [&disp_img](uchar& color, const int* pos) -> void {
            float disp = disp_img.at<float>(pos[0], pos[1]);
            color = uchar(255. *fmax(fmin(1.0, disp), 0.0));
        }
    );

    // output.create(disp_img.size(), CV_8UC3);
    // cv::applyColorMap(gray_scale, output, cv::COLORMAP_JET);
}

void pyramid_main(int argc, char** argv) {
    if (argc > 1) {
        timer = std::unique_ptr<TicToc>(new TicToc);
    }
    cv::Mat prev_img = cv::imread("../left.bmp");
    cv::Mat next_img = cv::imread("../right.bmp");
    
    printf("Prev size: (%d, %d), next size: (%d, %d)\n", prev_img.cols, prev_img.rows, next_img.cols, prev_img.rows);
    int rows = prev_img.rows, cols = prev_img.cols;
    border_padding_size(rows, cols, 8);
    rows -= prev_img.rows;
    cols -= prev_img.cols;
    cv::copyMakeBorder(prev_img, prev_img, 0, rows, 0, cols, cv::BORDER_REFLECT);
    cv::copyMakeBorder(next_img, next_img, 0, rows, 0, cols, cv::BORDER_REFLECT);
    printf("Prev size: (%d, %d), next size: (%d, %d)\n", prev_img.cols, prev_img.rows, next_img.cols, prev_img.rows);
    cv::Mat out = prev_img.clone();
    cv::Mat arrow = prev_img.clone();
    cv::Mat r2l = pyramid_searching(next_img, prev_img, arrow, out, 4, 4, true);
    cv::Mat l2r = pyramid_searching(prev_img, next_img, arrow, out, 4, 4, true);

    cv::Mat gray_disp_l2r, gray_disp_r2l;
    get_disparity_image(l2r, gray_disp_l2r);
    get_disparity_image(l2r, gray_disp_r2l);

    gray_disp_l2r.forEach<uchar>(
        [&gray_disp_r2l](uchar& color, const int* pos) -> void {
            uchar other = gray_disp_r2l.at<uchar>(pos[0], pos[1]);
            color = (color >= other) ? color : other;
        }
    );

    cv::Mat colored_disp, filtered_disp;
    cv::applyColorMap(gray_disp_l2r, colored_disp, cv::COLORMAP_JET);
    cv::medianBlur(colored_disp, filtered_disp, 5);
    cv::bilateralFilter(filtered_disp, colored_disp, 5, 20, 10);

    float psnr = image_psnr(out, next_img);
    printf("PSNR: %f\n", psnr);
    cv::imshow("disp", colored_disp);
    // cv::imshow("error", next_img - out);
    // cv::imshow("motion", arrow);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    cv::setNumThreads(2);
    pyramid_main(argc, argv);
    cv::destroyAllWindows();
    return 0;
}
