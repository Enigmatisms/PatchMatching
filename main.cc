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

void multi_step_main(int argc, char** argv) {
    int max_range = 32;
    int max_radius = 4, pad_size = max_range + max_radius;
    if (argc < 4) {
        printf("TSS usage: ./Task <mode> <max search range> <patch radius> <use timer? any: yes, nothing: no>\n");
    } else {
        max_range = atoi(argv[2]);
        max_radius = atoi(argv[3]);
        if (argc > 4) {
            timer = std::unique_ptr<TicToc>(new TicToc);
        }
    }
    cv::Mat prev_img = cv::imread("../prev.bmp");
    cv::Mat next_img = cv::imread("../next.bmp");
    cv::resize(prev_img, prev_img, prev_img.size() * 2);
    cv::resize(next_img, next_img, next_img.size() * 2);
    cv::Mat out = prev_img.clone();
    cv::Mat original_next = next_img.clone();
    cv::Mat arrow(prev_img.rows, prev_img.cols, CV_8UC3);
    cv::copyMakeBorder(prev_img, prev_img, pad_size, pad_size, pad_size, pad_size, cv::BORDER_REFLECT);
    cv::copyMakeBorder(next_img, next_img, pad_size, pad_size, pad_size, pad_size, cv::BORDER_REFLECT);
    
    multi_step_searching(prev_img, next_img, arrow, out, max_range, max_radius);

    float psnr = image_psnr(out, original_next);
    printf("PSNR: %f\n", psnr);
    cv::imshow("predict", out);
    cv::imshow("error", original_next - out);
    cv::imshow("motion", arrow);
    cv::waitKey(0);
}

void exhaustive_main(int argc, char** argv) {
    if (argc > 2) {
        timer = std::unique_ptr<TicToc>(new TicToc);
    }
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

    exhaustive_search(prev_img, next_img, arrow, out, patch_size);

    float psnr = image_psnr(out, next_img);
    printf("PSNR: %f\n", psnr);
    cv::imshow("predict", out);
    cv::imshow("error", next_img - out);
    cv::imshow("motion", arrow);
    cv::waitKey(0);
}

void pyramid_main(int argc, char** argv) {
    if (argc > 2) {
        timer = std::unique_ptr<TicToc>(new TicToc);
    }
    cv::Mat prev_img = cv::imread("../prev.bmp");
    cv::Mat next_img = cv::imread("../next.bmp");
    
    printf("Prev size: (%d, %d), next size: (%d, %d)\n", prev_img.cols, prev_img.rows, next_img.cols, prev_img.rows);
    cv::resize(prev_img, prev_img, prev_img.size() * 2);
    cv::resize(next_img, next_img, next_img.size() * 2);
    int rows = prev_img.rows, cols = prev_img.cols;
    border_padding_size(rows, cols, 8);
    rows -= prev_img.rows;
    cols -= prev_img.cols;
    cv::copyMakeBorder(prev_img, prev_img, 0, rows, 0, cols, cv::BORDER_REFLECT);
    cv::copyMakeBorder(next_img, next_img, 0, rows, 0, cols, cv::BORDER_REFLECT);
    printf("Prev size: (%d, %d), next size: (%d, %d)\n", prev_img.cols, prev_img.rows, next_img.cols, prev_img.rows);
    cv::Mat out = prev_img.clone();
    cv::Mat arrow = prev_img.clone();
    pyramid_searching(prev_img, next_img, arrow, out, 4);

    float psnr = image_psnr(out, next_img);
    printf("PSNR: %f\n", psnr);
    cv::imshow("predict", out);
    cv::imshow("error", next_img - out);
    cv::imshow("motion", arrow);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    cv::setNumThreads(2);
    if (argc < 2) {
        printf("Usage: ./Task <mode: 0 EMBA, 1: TSS(multi-step, not 3 steps), 2: HMBA>\n");
        return 1;
    } else {
        switch (atoi(argv[1])) {
            case 0: {exhaustive_main(argc, argv); break;}
            case 1: {multi_step_main(argc, argv); break;}
            case 2: {pyramid_main(argc, argv); break;}
            default: {pyramid_main(argc, argv); break;}
        };
        cv::destroyAllWindows();
    }
    return 0;
}
