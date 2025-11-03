#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include <opencv2/opencv.hpp>
#include <vector>

class FeatureMatcher
{
public:
    // 构造：可以指定 ORB 特征数与 Lowe ratio
    FeatureMatcher(int nFeatures = 4000, float ratio = 0.75f);

    // 对两张图片做特征检测、描述子计算与匹配（Lowe Ratio Test）
    // 输入：img1, img2（彩色或灰度均可）
    // 输出：kps1,kps2（关键点），good_matches（过滤后的匹配），outImg（可视化图像，若 draw=false 则为空）
    // 返回：匹配成功并且 good_matches 非空返回 true，否则返回 false
    bool matchImages(
        const cv::Mat& img1,
        const cv::Mat& img2,
        std::vector<cv::KeyPoint>& kps1,
        std::vector<cv::KeyPoint>& kps2,
        std::vector<cv::DMatch>& good_matches,
        cv::Mat& outImg,
        bool draw = true);

    // 设置 ratio
    void setRatio(float r) { ratio_ = r; }

private:
    int nFeatures_;
    float ratio_;
    cv::Ptr<cv::ORB> orb_;
};

#endif // FEATURE_MATCHER_H
#pragma once
