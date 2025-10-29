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
    // ���캯����ָ�� ORB ������������ Lowe ratio
    FeatureMatcher(int nFeatures = 4000, float ratio = 0.75f);

    // ��ȡ��������������
    bool extractFeatures(
        const cv::Mat& img,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors);

    // ƥ������ͼ�����������ֱ�����ƥ�������
    // ���룺����ͼ�����������������
    // �����old_match_points, new_match_points����Ӧ����㣩
    bool matchFeatures(
        const std::vector<cv::KeyPoint>& kps1,
        const cv::Mat& desc1,
        const std::vector<cv::KeyPoint>& kps2,
        const cv::Mat& desc2,
        std::vector<cv::Point2d>& old_match_points,
        std::vector<cv::Point2d>& new_match_points);

    // ���� Ratio
    void setRatio(float r) { ratio_ = r; }

private:
    int nFeatures_;
    float ratio_;
    cv::Ptr<cv::ORB> orb_;
};