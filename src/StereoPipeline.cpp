#include "StereoPipeline.h"
#include <iostream>

bool RunTwoViewReconstruction(const std::vector<cv::Point2d>& pts1,
                              const std::vector<cv::Point2d>& pts2,
                              const cv::Mat& K,
                              const cv::Mat& distCoeffs,
                              const std::string& outPly,
                              cv::Mat& R,
                              cv::Mat& t,
                              std::vector<cv::Point3d>& points3D_out,
                              double ransacThresh,
                              double reprojThreshold)
{
    points3D_out.clear();
    if (pts1.size() < 5 || pts2.size() < 5 || pts1.size() != pts2.size()) {
        std::cerr << "输入匹配点不足或长度不一致。" << std::endl;
        return false;
    }

    // 1) 计算本质矩阵（RANSAC）
    cv::Mat E, essentialMask;
    if (!ComputeEssentialMatrix(pts1, pts2, K, E, essentialMask, ransacThresh)) {
        std::cerr << "ComputeEssentialMatrix 失败。" << std::endl;
        return false;
    }

    // 收集 findEssentialMat 的内点
    std::vector<cv::Point2d> e_pts1, e_pts2;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (!essentialMask.empty() && essentialMask.type() == CV_8U) {
            if (essentialMask.at<uchar>((int)i) == 0) continue;
        }
        e_pts1.push_back(pts1[i]);
        e_pts2.push_back(pts2[i]);
    }

    if (e_pts1.size() < 5) {
        std::cerr << "Essential 内点过少：" << e_pts1.size() << std::endl;
        return false;
    }

    // 2) 从 E 恢复相对位姿（recoverPose 同时返回内点掩码）
    cv::Mat recoverMask;
    if (!RecoverPoseFromEssential(E, e_pts1, e_pts2, K, R, t, recoverMask)) {
        std::cerr << "RecoverPoseFromEssential 失败。" << std::endl;
        return false;
    }

    // 3)三角化点
    std::vector<cv::Point2d> tri_pts1, tri_pts2;
    for (size_t i = 0; i < e_pts1.size(); ++i) {
        if (!recoverMask.empty() && recoverMask.type() == CV_8U) {
            if (recoverMask.at<uchar>((int)i) == 0) continue;
        }
        tri_pts1.push_back(e_pts1[i]);
        tri_pts2.push_back(e_pts2[i]);
    }

    if (tri_pts1.size() < 2) {
        std::cerr << "用于三角化的内点过少：" << tri_pts1.size() << std::endl;
        return false;
    }

    // 4) 构建投影矩阵 P1, P2（相机1 为参考）
    cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat zero = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat P1, P2;
    cv::hconcat(I, zero, P1); P1 = K * P1;
    cv::Mat R64, t64;
    R.convertTo(R64, CV_64F);
    t.convertTo(t64, CV_64F);
    cv::hconcat(R64, t64, P2); P2 = K * P2;

    // 5) 三角化并筛选（深度与重投影）
    std::vector<unsigned char> validMask;
    std::vector<cv::Point3d> triPoints;
    if (!TriangulateTwoViews(P1, P2, tri_pts1, tri_pts2, triPoints, validMask, reprojThreshold)) {
        std::cerr << "TriangulateTwoViews 未产生有效点。" << std::endl;
        return false;
    }

    points3D_out = triPoints;

    // 6) 保存 PLY（若指定文件名）
    if (!outPly.empty()) {
        if (!SavePointCloudPLY(outPly, points3D_out)) {
            std::cerr << "SavePointCloudPLY 保存失败：" << outPly << std::endl;
            return false;
        }
    }

    return !points3D_out.empty();
}