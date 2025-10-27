#include "FeatureMatcher.h"
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

FeatureMatcher::FeatureMatcher(int nFeatures, float ratio)
    : nFeatures_(nFeatures), ratio_(ratio)
{
    orb_ = cv::ORB::create(nFeatures_);
}

// 主匹配函数实现
bool FeatureMatcher::matchImages(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::KeyPoint>& kps1,
    std::vector<cv::KeyPoint>& kps2,
    std::vector<cv::DMatch>& good_matches,
    cv::Mat& outImg,
    bool draw)
{
    good_matches.clear();
    outImg.release();

    if (img1.empty() || img2.empty()) {
        std::cerr << "[FeatureMatcher] 输入图片为空！" << std::endl;
        return false;
    }

    // 转灰度（ORB 要求单通道）
    cv::Mat gray1, gray2;
    if (img1.channels() == 3) cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    else gray1 = img1;
    if (img2.channels() == 3) cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    else gray2 = img2;

    // 检测并计算描述子
    cv::Mat desc1, desc2;
    orb_->detectAndCompute(gray1, cv::noArray(), kps1, desc1);
    orb_->detectAndCompute(gray2, cv::noArray(), kps2, desc2);

    if (desc1.empty() || desc2.empty() || kps1.empty() || kps2.empty()) {
        std::cerr << "[FeatureMatcher] 未能检测到足够特征点或描述子为空。" << std::endl;
        return false;
    }

    // BFMatcher + Hamming，因为 ORB 使用二进制描述子
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2); // 每个描述子取两个最近邻

    // Lowe ratio test 过滤
    for (size_t i = 0; i < knn_matches.size(); ++i) {
        if (knn_matches[i].size() >= 2) {
            const cv::DMatch& m1 = knn_matches[i][0];
            const cv::DMatch& m2 = knn_matches[i][1];
            if (m1.distance < ratio_ * m2.distance) {
                good_matches.push_back(m1);
            }
        }
    }

    if (good_matches.empty()) {
        std::cerr << "[FeatureMatcher] 经过 ratio test 后无匹配。" << std::endl;
        return false;
    }

    // 可视化匹配（使用 OpenCV 的 drawMatches，颜色为默认随机配色）
    if (draw) {
        // 为了看得更清楚，画出匹配线并显示关键点
        cv::drawMatches(img1, kps1, img2, kps2, good_matches, outImg,
            cv::Scalar::all(-1), cv::Scalar::all(-1),
            std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }

    return true;
}
