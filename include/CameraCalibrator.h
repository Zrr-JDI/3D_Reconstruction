#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * @brief 相机标定结果结构体
 */
struct CalibrationResult {
    bool success;
    std::vector<cv::Mat> Ks;           // 内参矩阵集合
    cv::Mat distCoeffs;                 // 畸变参数
    double reprojectionError;           // 重投影误差
    std::string errorMessage;
    
    CalibrationResult() : success(false), reprojectionError(0.0) {}
};

/**
 * @brief 相机标定器类
 */
class MonoCameraCalibrator {
public:
    /**
     * @brief 构造函数
     * @param boardSize 棋盘格内角点数量
     * @param squareSize 棋盘格方格实际尺寸(米)
     */
    MonoCameraCalibrator(cv::Size boardSize, float squareSize);
    
    /**
     * @brief 执行完整的相机标定流程
     * @param imagePaths 棋盘格图像路径列表
     * @return 标定结果
     */
    CalibrationResult calibrateCamera(const std::vector<std::string>& imagePaths);
    
    /**
     * @brief 计算相机内参矩阵 K
     * @param objectPoints 世界坐标系中的角点
     * @param imagePoints 图像坐标系中的角点
     * @param imageSize 图像尺寸
     * @param Ks 输出: 内参矩阵集合
     * @return 计算是否成功
     */
    bool computeCameraIntrinsics(const std::vector<std::vector<cv::Point3f>>& objectPoints,
                                const std::vector<std::vector<cv::Point2f>>& imagePoints,
                                cv::Size imageSize, std::vector<cv::Mat>& Ks);
    
    /**
     * @brief 计算畸变参数
     * @param objectPoints 世界坐标系中的角点
     * @param imagePoints 图像坐标系中的角点
     * @param imageSize 图像尺寸
     * @param cameraMatrix 内参矩阵
     * @param distCoeffs 输出: 畸变参数
     * @param reprojectionError 输出: 重投影误差
     * @return 计算是否成功
     */
    bool computeDistortionCoefficients(const std::vector<std::vector<cv::Point3f>>& objectPoints,
                                     const std::vector<std::vector<cv::Point2f>>& imagePoints,
                                     cv::Size imageSize, const cv::Mat& cameraMatrix,
                                     cv::Mat& distCoeffs, double& reprojectionError);
    
    /**
     * @brief 获取标定状态信息
     * @return 状态描述字符串
     */
    std::string getStatus() const;

private:
    cv::Size m_boardSize;      // 棋盘格角点数量
    float m_squareSize;        // 棋盘格方格实际尺寸
    std::string m_status;      // 内部状态信息
    
    // 查找棋盘格角点
    bool findChessboardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners);
    
    // 生成世界坐标系中的角点
    void generateObjectPoints(std::vector<cv::Point3f>& objectPoints);
    
    // 验证标定结果
    bool validateCalibration(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, double reprojectionError);
    
    // 更新状态信息
    void updateStatus(const std::string& message);
};