#include <iostream>
#include <vector>
#include <string>
#include <fstream>

// OpenCV 相关头文件
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp> 
#include <opencv2/sfm.hpp>

// 对于 Point3d 和 Point2d 类型
#include <opencv2/core/types.hpp>


bool PnP(std::vector<cv::Point3d>& model_points, std::vector<cv::Point2d>& image_points, cv::Mat& camera_matrix, cv::Mat& diffcoeffs, cv::Mat& translate_vec, cv::Mat& rotate_vec);

bool Bundle_Adjustment(const std::vector<cv::Mat>& points2d_all, const std::vector<cv::Mat>& points3d_all, const std::vector<cv::Mat>& K, const std::vector<cv::Mat>& Rs, const std::vector<cv::Mat>& ts);

cv::Vec4d rotationMatrixToQuaternion(const cv::Mat& R);

void Export_To_NVM(
    const std::vector<std::string>& imageNames,
    const std::vector<cv::Mat>& Rs,
    const std::vector<cv::Mat>& ts,
    const std::vector<cv::Mat>& Ks,
    const std::vector<cv::Point3d>& points3D,
    const std::vector<std::vector<cv::Point2d>>& projections2D_all, 
    const std::vector<std::vector<int>>& viewIndices,
    const std::vector<cv::Mat>& images
);

bool IncreaseCloud(std::vector<std::vector<int>>& point3DIds,std::vector<std::vector<cv::Point2d>>& feature_points,int camera_number,int num, cv::Mat& old_camera, std::vector<cv::Point2d>& old_image_points, std::vector<cv::Point3d>& model_points, std::vector<cv::Point2d>& image_points, cv::Mat& camera_matrix, cv::Mat& diffcoeffs, cv::Mat& translate_vec, cv::Mat& rotate_vec, std::vector<std::vector<int>>& viewIndices);