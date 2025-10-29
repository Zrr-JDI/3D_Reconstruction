#pragma once
#ifndef SPARSETRIANGULATION_H
#define SPARSETRIANGULATION_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// 在两个视图间三角化并做正深度 + 重投影误差筛选
// P1, P2: 3x4 投影矩阵（已包含内参 K）
// pts1, pts2: 对应的像素坐标向量（同长度）
// points3D: 输出通过筛选的三维点（相机1参考系）
// validMask: 输出与输入点等长的 mask（1: 有效，0: 无效）
// reprojThreshold: 重投影误差阈值（像素）
// 返回 true 表示至少有一个有效点
bool TriangulateTwoViews(const cv::Mat& P1, const cv::Mat& P2,
                         const std::vector<cv::Point2d>& pts1,
                         const std::vector<cv::Point2d>& pts2,
                         std::vector<cv::Point3d>& points3D,
                         std::vector<unsigned char>& validMask,
                         double reprojThreshold = 3.0);

// 将稀疏点云写入 ASCII PLY（仅顶点）
// 返回 true 表示写入成功
bool SavePointCloudPLY(const std::string& filename,
                       const std::vector<cv::Point3d>& points3D);

#endif // SPARSETRIANGULATION_H