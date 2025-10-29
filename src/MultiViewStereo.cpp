#include "MultiViewStereo.h"
#include "SparseTriangulation.h"
#include <cmath>



bool ComputeEssentialMatrix(const std::vector<cv::Point2d>& pts1,
                            const std::vector<cv::Point2d>& pts2,
                            const cv::Mat& K,
                            cv::Mat& E,
                            cv::Mat& inlierMask,
                            double ransacThresh)
{
    if (pts1.size() < 5 || pts2.size() < 5 || pts1.size() != pts2.size())
        return false;

    std::vector<cv::Point2f> p1f, p2f;
    p1f.reserve(pts1.size()); p2f.reserve(pts2.size());
    for (size_t i = 0; i < pts1.size(); ++i) {
        p1f.emplace_back(static_cast<float>(pts1[i].x), static_cast<float>(pts1[i].y));
        p2f.emplace_back(static_cast<float>(pts2[i].x), static_cast<float>(pts2[i].y));
    }

    // findEssentialMat �᷵�� 3x3 ������⣨�ڷֿ�����£�����д������
    E = cv::findEssentialMat(p1f, p2f, K, cv::RANSAC, 0.999, ransacThresh, inlierMask);

    return !E.empty();
}

bool RecoverPoseFromEssential(const cv::Mat& E,
                              const std::vector<cv::Point2d>& pts1,
                              const std::vector<cv::Point2d>& pts2,
                              const cv::Mat& K,
                              cv::Mat& R,
                              cv::Mat& t,
                              cv::Mat& inlierMask)
{
    if (E.empty()) return false;
    if (pts1.size() != pts2.size() || pts1.size() < 5) return false;

    std::vector<cv::Point2f> p1f, p2f;
    p1f.reserve(pts1.size()); p2f.reserve(pts2.size());
    for (size_t i = 0; i < pts1.size(); ++i) {
        p1f.emplace_back(static_cast<float>(pts1[i].x), static_cast<float>(pts1[i].y));
        p2f.emplace_back(static_cast<float>(pts2[i].x), static_cast<float>(pts2[i].y));
    }

    // recoverPose ����� R, t��t ��λ�����������ڵ�����ͬʱ��д������
    int inliers = cv::recoverPose(E, p1f, p2f, K, R, t, inlierMask);
    return inliers > 0;
}