#include "CameraCalibrator.h"
#include <iostream>
#include <fstream>
#include <chrono>

// 构造函数：初始化棋盘格尺寸和方格实际大小
MonoCameraCalibrator::MonoCameraCalibrator(cv::Size boardSize, float squareSize)
    : m_boardSize(boardSize), m_squareSize(squareSize) {
    updateStatus("相机标定器初始化完成");
}

// 执行完整的相机标定流程
CalibrationResult MonoCameraCalibrator::calibrateCamera(const std::vector<std::string>& imagePaths) {
    CalibrationResult result;  // 创建结果对象
    auto startTime = std::chrono::high_resolution_clock::now();  // 记录开始时间

    updateStatus("开始相机标定流程...");

    // 检查输入图像路径是否为空
    if (imagePaths.empty()) {
        result.errorMessage = "图像路径列表为空";
        updateStatus("标定失败: " + result.errorMessage);
        return result;
    }

    // 准备存储世界坐标点和图像坐标点的容器
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Size imageSize(0, 0);  // 图像尺寸初始化为0
    int validImageCount = 0;   // 有效图像计数器

    updateStatus("检测棋盘格角点...");

    // 1. 遍历所有图像，检测棋盘格角点
    for (size_t i = 0; i < imagePaths.size(); i++) {
        // 读取图像（灰度模式）
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            updateStatus("警告: 无法读取图像: " + imagePaths[i]);
            continue;
        }

        // 记录第一张有效图像的尺寸
        if (imageSize.width == 0) {
            imageSize = image.size();
        }

        // 检测棋盘格角点
        std::vector<cv::Point2f> corners;
        if (findChessboardCorners(image, corners)) {
            // 生成对应的世界坐标系3D点
            std::vector<cv::Point3f> objPoints;
            generateObjectPoints(objPoints);

            // 存储检测到的点
            objectPoints.push_back(objPoints);
            imagePoints.push_back(corners);
            validImageCount++;

            updateStatus("成功检测图像 " + std::to_string(i + 1) + " 的角点");
        }
        else {
            updateStatus("警告: 未能在图像 " + std::to_string(i + 1) + " 中找到棋盘格角点");
        }
    }

    updateStatus("有效图像数量: " + std::to_string(validImageCount) + "/" + std::to_string(imagePaths.size()));

    // 检查是否有足够数量的有效图像
    if (validImageCount < 5) {
        result.errorMessage = "有效图像数量不足，至少需要5张有效的棋盘格图像";
        updateStatus("标定失败: " + result.errorMessage);
        return result;
    }

    // 检查是否成功获取图像尺寸
    if (imageSize.width == 0) {
        result.errorMessage = "无法确定图像尺寸";
        updateStatus("标定失败: " + result.errorMessage);
        return result;
    }

    // 2. 计算相机内参矩阵 K
    updateStatus("计算相机内参矩阵 K...");
    std::vector<cv::Mat> Ks;
    if (!computeCameraIntrinsics(objectPoints, imagePoints, imageSize, Ks)) {
        result.errorMessage = "计算相机内参矩阵失败";
        updateStatus("标定失败: " + result.errorMessage);
        return result;
    }

    // 3. 计算畸变参数
    updateStatus("计算畸变参数...");
    cv::Mat distCoeffs;
    double reprojectionError = 0.0;
    if (!computeDistortionCoefficients(objectPoints, imagePoints, imageSize, Ks[0],
        distCoeffs, reprojectionError)) {
        result.errorMessage = "计算畸变参数失败";
        updateStatus("标定失败: " + result.errorMessage);
        return result;
    }

    // 4. 验证标定结果
    if (validateCalibration(Ks[0], distCoeffs, reprojectionError)) {
        result.success = true;
        result.Ks = Ks;
        result.distCoeffs = distCoeffs;
        result.reprojectionError = reprojectionError;

        // 计算并显示标定耗时
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        updateStatus("相机标定成功完成! 重投影误差: " +
            std::to_string(reprojectionError) + " 像素, 耗时: " +
            std::to_string(duration.count()) + "ms");
    }
    else {
        result.errorMessage = "标定结果验证失败";
        updateStatus("标定失败: " + result.errorMessage);
    }

    return result;
}

// 计算相机内参矩阵
bool MonoCameraCalibrator::computeCameraIntrinsics(const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    cv::Size imageSize, std::vector<cv::Mat>& Ks) {
    // 初始化相机矩阵为单位矩阵
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

    // 初始化畸变系数和旋转/平移向量
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;

    // 使用OpenCV的calibrateCamera函数计算内参
    double reprojectionError = cv::calibrateCamera(
        objectPoints, imagePoints, imageSize,
        cameraMatrix, distCoeffs,
        rvecs, tvecs,
        cv::CALIB_FIX_ASPECT_RATIO |  // 固定宽高比
        cv::CALIB_ZERO_TANGENT_DIST | // 忽略切向畸变
        cv::CALIB_FIX_K3              // 固定k3畸变系数
    );

    // 检查内参矩阵的有效性
    if (cameraMatrix.empty() || cameraMatrix.at<double>(0, 0) <= 0) {
        updateStatus("错误: 计算的内参矩阵无效");
        return false;
    }

    // 将内参矩阵添加到结果向量中
    Ks.clear();
    Ks.push_back(cameraMatrix);

    // 打印内参矩阵的关键参数
    updateStatus("相机内参矩阵计算成功");
    updateStatus("焦距 fx: " + std::to_string(cameraMatrix.at<double>(0, 0)));
    updateStatus("焦距 fy: " + std::to_string(cameraMatrix.at<double>(1, 1)));
    updateStatus("主点 cx: " + std::to_string(cameraMatrix.at<double>(0, 2)));
    updateStatus("主点 cy: " + std::to_string(cameraMatrix.at<double>(1, 2)));

    return true;
}

// 计算畸变参数
bool MonoCameraCalibrator::computeDistortionCoefficients(const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    cv::Size imageSize, const cv::Mat& cameraMatrix,
    cv::Mat& distCoeffs, double& reprojectionError) {
    // 初始化畸变系数
    distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

    // 使用OpenCV的calibrateCamera函数计算畸变参数
    std::vector<cv::Mat> rvecs, tvecs;

    reprojectionError = cv::calibrateCamera(
        objectPoints, imagePoints, imageSize,
        const_cast<cv::Mat&>(cameraMatrix), distCoeffs,
        rvecs, tvecs,
        cv::CALIB_USE_INTRINSIC_GUESS | // 使用已知内参作为初始值
        cv::CALIB_FIX_ASPECT_RATIO |    // 固定宽高比
        cv::CALIB_ZERO_TANGENT_DIST |   // 忽略切向畸变
        cv::CALIB_FIX_K3                // 固定k3畸变系数
    );

    // 检查畸变参数的有效性
    if (distCoeffs.empty()) {
        updateStatus("错误: 计算的畸变参数无效");
        return false;
    }

    // 打印畸变参数和重投影误差
    updateStatus("畸变参数计算成功");
    updateStatus("径向畸变 k1: " + std::to_string(distCoeffs.at<double>(0, 0)));
    updateStatus("径向畸变 k2: " + std::to_string(distCoeffs.at<double>(0, 1)));
    updateStatus("切向畸变 p1: " + std::to_string(distCoeffs.at<double>(0, 2)));
    updateStatus("切向畸变 p2: " + std::to_string(distCoeffs.at<double>(0, 3)));
    updateStatus("径向畸变 k3: " + std::to_string(distCoeffs.at<double>(0, 4)));
    updateStatus("重投影误差: " + std::to_string(reprojectionError) + " 像素");

    return true;
}

// 在图像中查找棋盘格角点
bool MonoCameraCalibrator::findChessboardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners) {
    // 使用OpenCV的findChessboardCorners函数检测角点
    bool found = cv::findChessboardCorners(image, m_boardSize, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

    if (found) {
        // 使用亚像素级角点精确化提高精度
        cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
    }

    return found;
}

// 生成世界坐标系中的棋盘格角点坐标
void MonoCameraCalibrator::generateObjectPoints(std::vector<cv::Point3f>& objectPoints) {
    objectPoints.clear();
    // 在Z=0平面上生成棋盘格角点的3D坐标
    for (int i = 0; i < m_boardSize.height; i++) {
        for (int j = 0; j < m_boardSize.width; j++) {
            objectPoints.push_back(cv::Point3f(j * m_squareSize, i * m_squareSize, 0));
        }
    }
}

// 验证标定结果的合理性
bool MonoCameraCalibrator::validateCalibration(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, double reprojectionError) {
    // 验证条件：内参矩阵非空、畸变参数非空、重投影误差小于2像素、焦距有效
    bool valid = !cameraMatrix.empty() &&
        !distCoeffs.empty() &&
        reprojectionError < 2.0 &&
        cameraMatrix.at<double>(0, 0) > 0;

    if (!valid) {
        updateStatus("标定结果验证失败: 重投影误差=" +
            std::to_string(reprojectionError));
    }

    return valid;
}

// 获取当前状态信息
std::string MonoCameraCalibrator::getStatus() const {
    return m_status;
}

// 更新状态信息并打印到控制台
void MonoCameraCalibrator::updateStatus(const std::string& message) {
    m_status = message;
    std::cout << "[CameraCalibrator] " << message << std::endl;
}