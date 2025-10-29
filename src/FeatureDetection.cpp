#include"FeatureDetection.h"


FeatureMatcher::FeatureMatcher(int nFeatures, float ratio)
    : nFeatures_(nFeatures), ratio_(ratio)
{
    orb_ = cv::ORB::create(nFeatures_);
}

// 提取特征点 + 描述子
bool FeatureMatcher::extractFeatures(
    const cv::Mat& img,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors)
{
    keypoints.clear();
    descriptors.release();

    if (img.empty()) {
        std::cerr << "[FeatureMatcher] 输入图像为空！" << std::endl;
        return false;
    }

    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;

    orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

    if (keypoints.empty() || descriptors.empty()) {
        std::cerr << "[FeatureMatcher] 未检测到特征点或描述子为空。" << std::endl;
        return false;
    }

    return true;
}

// 匹配特征点并输出对应坐标（Point2d）
bool FeatureMatcher::matchFeatures(
    const std::vector<cv::KeyPoint>& kps1,
    const cv::Mat& desc1,
    const std::vector<cv::KeyPoint>& kps2,
    const cv::Mat& desc2,
    std::vector<cv::Point2d>& old_match_points,
    std::vector<cv::Point2d>& new_match_points)
{
    old_match_points.clear();
    new_match_points.clear();

    if (desc1.empty() || desc2.empty() || kps1.empty() || kps2.empty()) {
        std::cerr << "[FeatureMatcher] 输入特征或描述子为空！" << std::endl;
        return false;
    }

    // 使用 Hamming 距离的暴力匹配器（适合 ORB）
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    // Lowe ratio test
    for (const auto& knn : knn_matches) {
        if (knn.size() < 2) continue;

        const cv::DMatch& m1 = knn[0];
        const cv::DMatch& m2 = knn[1];

        if (m1.distance < ratio_ * m2.distance) {
            // 提取匹配到的坐标点
            cv::Point2d p1 = kps1[m1.queryIdx].pt;
            cv::Point2d p2 = kps2[m1.trainIdx].pt;
            old_match_points.push_back(p1);
            new_match_points.push_back(p2);
        }
    }

    if (old_match_points.empty() || new_match_points.empty()) {
        std::cerr << "[FeatureMatcher] ratio test 后无有效匹配。" << std::endl;
        return false;
    }

    return true;
}