#include "StereoPipeline.h"
#include <iostream>

bool RunTwoViewReconstruction(
    const std::vector<cv::Point2d>& pts1,
    const std::vector<cv::Point2d>& pts2,
    const cv::Mat& K,
    const cv::Mat& distCoeffs,
    const std::string& outPly,
    cv::Mat& R,
    cv::Mat& t,
    std::vector<cv::Point3d>& points3D_out,
    std::vector<cv::Point3d>& globalPoints3D,                 // ȫ��3D������
    std::vector<std::vector<cv::Point2d>>& projections2D_all, // ÿ�������ͶӰ��
    std::vector<std::vector<int>>& point3DIds,                // ÿ��ͼ���������Ӧ��3D��ID
    double ransacThresh,
    double reprojThreshold)
{
    points3D_out.clear();

    if (pts1.size() < 5 || pts2.size() < 5 || pts1.size() != pts2.size()) {
        std::cerr << "����ƥ��㲻��򳤶Ȳ�һ�¡�" << std::endl;
        return false;
    }

    // 1) ���㱾�ʾ���
    cv::Mat E, essentialMask;
    if (!ComputeEssentialMatrix(pts1, pts2, K, E, essentialMask, ransacThresh)) {
        std::cerr << "ComputeEssentialMatrix ʧ�ܡ�" << std::endl;
        return false;
    }

    // �������ڵ�
    std::vector<cv::Point2d> e_pts1, e_pts2;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (!essentialMask.empty() && essentialMask.at<uchar>((int)i) == 0) continue;
        e_pts1.push_back(pts1[i]);
        e_pts2.push_back(pts2[i]);
    }

    if (e_pts1.size() < 5) {
        std::cerr << "Essential �ڵ���٣�" << e_pts1.size() << std::endl;
        return false;
    }

    // 2) �� E �ָ����λ�� 
    cv::Mat recoverMask;
    if (!RecoverPoseFromEssential(E, e_pts1, e_pts2, K, R, t, recoverMask)) {
        std::cerr << "RecoverPoseFromEssential ʧ�ܡ�" << std::endl;
        return false;
    }

    // ������ pose �ڵ�
    std::vector<cv::Point2d> tri_pts1, tri_pts2;
    for (size_t i = 0; i < e_pts1.size(); ++i) {
        if (!recoverMask.empty() && recoverMask.at<uchar>((int)i) == 0) continue;
        tri_pts1.push_back(e_pts1[i]);
        tri_pts2.push_back(e_pts2[i]);
    }

    if (tri_pts1.size() < 2) {
        std::cerr << "�������ǻ����ڵ���٣�" << tri_pts1.size() << std::endl;
        return false;
    }

    // 3) ����ͶӰ���� P1, P2 
    cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat zero = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat P1, P2;
    cv::hconcat(I, zero, P1); P1 = K * P1;

    cv::Mat R64, t64;
    R.convertTo(R64, CV_64F);
    t.convertTo(t64, CV_64F);
    cv::hconcat(R64, t64, P2); P2 = K * P2;

    // 4) ���ǻ� 
    std::vector<unsigned char> validMask;
    std::vector<cv::Point3d> triPoints;
    if (!TriangulateTwoViews(P1, P2, tri_pts1, tri_pts2, triPoints, validMask, reprojThreshold)) {
        std::cerr << "TriangulateTwoViews δ������Ч�㡣" << std::endl;
        return false;
    }

    points3D_out = triPoints;

    // 5) ����ȫ������ 
    int baseIdx = (int)globalPoints3D.size();
    globalPoints3D.insert(globalPoints3D.end(), triPoints.begin(), triPoints.end());

    // projections2D_all[0] ��Ӧ���1�� projections2D_all[1] ��Ӧ���2
    if (projections2D_all.size() < 2) {
        projections2D_all.resize(2);
        point3DIds.resize(2);
    }

    for (size_t i = 0; i < tri_pts1.size(); ++i) {
        projections2D_all[0].push_back(tri_pts1[i]);
        projections2D_all[1].push_back(tri_pts2[i]);
        point3DIds[0].push_back(baseIdx + (int)i);
        point3DIds[1].push_back(baseIdx + (int)i);
    }

    // 6) ����PLY 
    if (!outPly.empty()) {
        if (!SavePointCloudPLY(outPly, points3D_out)) {
            std::cerr << "SavePointCloudPLY ����ʧ�ܣ�" << outPly << std::endl;
            return false;
        }
    }

    return !points3D_out.empty();
}
