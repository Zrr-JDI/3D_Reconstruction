#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp> 
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

class FeatureMatcher
{
public:
    // 构造函数：指定 ORB 特征点数量与 Lowe ratio
    FeatureMatcher(int nFeatures = 4000, float ratio = 0.75f);

    // 提取特征点与描述子
    bool extractFeatures(
        const cv::Mat& img,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors);

    // 匹配两张图像的特征，并直接输出匹配坐标点
    // 输入：两张图像的特征点与描述子
    // 输出：old_match_points, new_match_points（对应坐标点）
    bool matchFeatures(
        const std::vector<cv::KeyPoint>& kps1,
        const cv::Mat& desc1,
        const std::vector<cv::KeyPoint>& kps2,
        const cv::Mat& desc2,
        std::vector<cv::Point2d>& old_match_points,
        std::vector<cv::Point2d>& new_match_points);

    // 设置 Ratio
    void setRatio(float r) { ratio_ = r; }

private:
    int nFeatures_;
    float ratio_;
    cv::Ptr<cv::ORB> orb_;
};