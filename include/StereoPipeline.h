#pragma once
#ifndef STEREOPIPELINE_H
#define STEREOPIPELINE_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MultiViewStereo.h"
#include "SparseTriangulation.h"

// 封装：对两视图执行 E 估计 -> 恢复 R,t -> 三角化 -> 保存稀疏点云
// 输入:
//   pts1, pts2 - 两张图像中的匹配像素点（对应顺序）
//   K, distCoeffs - 相机内参与畸变（distCoeffs 目前未用于去畸变，保留参数以便扩展）
//   outPly - 输出 PLY 文件名（若为空则不保存）
// 输出:
//   R, t - 从视图1到视图2 的相对旋转与平移（t 单位化方向）
//   points3D_out - 生成的稀疏三维点（相机1 参考系）
// 返回:
//   true 表示整个流程成功并至少产生一个3D点（若 outPly 非空则写文件成功）
bool RunTwoViewReconstruction(const std::vector<cv::Point2d>& pts1,
                              const std::vector<cv::Point2d>& pts2,
                              const cv::Mat& K,
                              const cv::Mat& distCoeffs,
                              const std::string& outPly,
                              cv::Mat& R,
                              cv::Mat& t,
                              std::vector<cv::Point3d>& points3D_out,
                              double ransacThresh = 1.0,
                              double reprojThreshold = 3.0);

#endif // STEREOPIPELINE_H