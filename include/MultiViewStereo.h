#pragma once
#ifndef MULTIVIEWSTEREO_H
#define MULTIVIEWSTEREO_H

#include <vector>
#include <opencv2/opencv.hpp>

// 计算本质矩阵（使用 RANSAC）
// pts1, pts2: 像素坐标（同一像素坐标系）
// K: 相机内参
// E: 输出本质矩阵
// inlierMask: 输出内点掩码 (CV_8U)
// ransacThresh: RANSAC 重投影阈值（像素）
bool ComputeEssentialMatrix(const std::vector<cv::Point2d>& pts1,
                            const std::vector<cv::Point2d>& pts2,
                            const cv::Mat& K,
                            cv::Mat& E,
                            cv::Mat& inlierMask,
                            double ransacThresh = 1.0);

// 从本质矩阵恢复相对姿态 R,t
// E: 本质矩阵
// pts1, pts2: 对应像素点（用于 recoverPose 的输入）
// K: 相机内参
// R, t: 输出相对旋转和平移（t 为单位向量方向）
// inlierMask: 可选，recoverPose 输出的内点掩码（CV_8U）
bool RecoverPoseFromEssential(const cv::Mat& E,
                              const std::vector<cv::Point2d>& pts1,
                              const std::vector<cv::Point2d>& pts2,
                              const cv::Mat& K,
                              cv::Mat& R,
                              cv::Mat& t,
                              cv::Mat& inlierMask);

#endif // MULTIVIEWSTEREO_H