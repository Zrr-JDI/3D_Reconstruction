#include"FeatureDetection.h"


FeatureMatcher::FeatureMatcher(int nFeatures, float ratio)
    : nFeatures_(nFeatures), ratio_(ratio)
{
    orb_ = cv::ORB::create(nFeatures_);
}

// ��ȡ������ + ������
bool FeatureMatcher::extractFeatures(
    const cv::Mat& img,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors)
{
    keypoints.clear();
    descriptors.release();

    if (img.empty()) {
        std::cerr << "[FeatureMatcher] ����ͼ��Ϊ�գ�" << std::endl;
        return false;
    }

    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;

    orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

    if (keypoints.empty() || descriptors.empty()) {
        std::cerr << "[FeatureMatcher] δ��⵽�������������Ϊ�ա�" << std::endl;
        return false;
    }

    return true;
}

// ƥ�������㲢�����Ӧ���꣨Point2d��
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
        std::cerr << "[FeatureMatcher] ����������������Ϊ�գ�" << std::endl;
        return false;
    }

    // ʹ�� Hamming ����ı���ƥ�������ʺ� ORB��
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    // Lowe ratio test
    for (const auto& knn : knn_matches) {
        if (knn.size() < 2) continue;

        const cv::DMatch& m1 = knn[0];
        const cv::DMatch& m2 = knn[1];

        if (m1.distance < ratio_ * m2.distance) {
            // ��ȡƥ�䵽�������
            cv::Point2d p1 = kps1[m1.queryIdx].pt;
            cv::Point2d p2 = kps2[m1.trainIdx].pt;
            old_match_points.push_back(p1);
            new_match_points.push_back(p2);
        }
    }

    if (old_match_points.empty() || new_match_points.empty()) {
        std::cerr << "[FeatureMatcher] ratio test ������Чƥ�䡣" << std::endl;
        return false;
    }

    return true;
}