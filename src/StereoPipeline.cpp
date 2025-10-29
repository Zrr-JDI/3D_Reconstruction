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
    std::vector<cv::Point3d>& globalPoints3D,                 // 全局3D点容器
    std::vector<std::vector<cv::Point2d>>& projections2D_all, // 每个相机的投影点
    std::vector<std::vector<int>>& point3DIds,                // 每个图像特征点对应的3D点ID
    double ransacThresh,
    double reprojThreshold)
{
    points3D_out.clear();

    if (pts1.size() < 5 || pts2.size() < 5 || pts1.size() != pts2.size()) {
        std::cerr << "输入匹配点不足或长度不一致。" << std::endl;
        return false;
    }

    // 1) 计算本质矩阵
    cv::Mat E, essentialMask;
    if (!ComputeEssentialMatrix(pts1, pts2, K, E, essentialMask, ransacThresh)) {
        std::cerr << "ComputeEssentialMatrix 失败。" << std::endl;
        return false;
    }

    // 仅保留内点
    std::vector<cv::Point2d> e_pts1, e_pts2;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (!essentialMask.empty() && essentialMask.at<uchar>((int)i) == 0) continue;
        e_pts1.push_back(pts1[i]);
        e_pts2.push_back(pts2[i]);
    }

    if (e_pts1.size() < 5) {
        std::cerr << "Essential 内点过少：" << e_pts1.size() << std::endl;
        return false;
    }

    // 2) 从 E 恢复相对位姿 
    cv::Mat recoverMask;
    if (!RecoverPoseFromEssential(E, e_pts1, e_pts2, K, R, t, recoverMask)) {
        std::cerr << "RecoverPoseFromEssential 失败。" << std::endl;
        return false;
    }

    // 仅保留 pose 内点
    std::vector<cv::Point2d> tri_pts1, tri_pts2;
    for (size_t i = 0; i < e_pts1.size(); ++i) {
        if (!recoverMask.empty() && recoverMask.at<uchar>((int)i) == 0) continue;
        tri_pts1.push_back(e_pts1[i]);
        tri_pts2.push_back(e_pts2[i]);
    }

    if (tri_pts1.size() < 2) {
        std::cerr << "用于三角化的内点过少：" << tri_pts1.size() << std::endl;
        return false;
    }

    // 3) 构建投影矩阵 P1, P2 
    cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat zero = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat P1, P2;
    cv::hconcat(I, zero, P1); P1 = K * P1;

    cv::Mat R64, t64;
    R.convertTo(R64, CV_64F);
    t.convertTo(t64, CV_64F);
    cv::hconcat(R64, t64, P2); P2 = K * P2;

    // 4) 三角化 
    std::vector<unsigned char> validMask;
    std::vector<cv::Point3d> triPoints;
    if (!TriangulateTwoViews(P1, P2, tri_pts1, tri_pts2, triPoints, validMask, reprojThreshold)) {
        std::cerr << "TriangulateTwoViews 未产生有效点。" << std::endl;
        return false;
    }

    points3D_out = triPoints;

    // 5) 保存全局数据 
    int baseIdx = (int)globalPoints3D.size();
    globalPoints3D.insert(globalPoints3D.end(), triPoints.begin(), triPoints.end());

    // projections2D_all[0] 对应相机1， projections2D_all[1] 对应相机2
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

    // 6) 保存PLY 
    if (!outPly.empty()) {
        if (!SavePointCloudPLY(outPly, points3D_out)) {
            std::cerr << "SavePointCloudPLY 保存失败：" << outPly << std::endl;
            return false;
        }
    }

    return !points3D_out.empty();
}
