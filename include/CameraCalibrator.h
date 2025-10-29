#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

/**
 * @brief ����궨����ṹ��
 */
struct CalibrationResult {
    bool success;
    std::vector<cv::Mat> Ks;           // �ڲξ��󼯺�
    cv::Mat distCoeffs;                 // �������
    double reprojectionError;           // ��ͶӰ���
    std::string errorMessage;

    CalibrationResult() : success(false), reprojectionError(0.0) {}
};

/**
 * @brief ����궨����
 */
class MonoCameraCalibrator {
public:
    /**
     * @brief ���캯��
     * @param boardSize ���̸��ڽǵ�����
     * @param squareSize ���̸񷽸�ʵ�ʳߴ�(��)
     */
    MonoCameraCalibrator(cv::Size boardSize, float squareSize);

    /**
     * @brief ִ������������궨����
     * @param imagePaths ���̸�ͼ��·���б�
     * @return �궨���
     */
    CalibrationResult calibrateCamera(const std::vector<std::string>& imagePaths);

    /**
     * @brief ��������ڲξ��� K
     * @param objectPoints ��������ϵ�еĽǵ�
     * @param imagePoints ͼ������ϵ�еĽǵ�
     * @param imageSize ͼ��ߴ�
     * @param Ks ���: �ڲξ��󼯺�
     * @return �����Ƿ�ɹ�
     */
    bool computeCameraIntrinsics(const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2f>>& imagePoints,
        cv::Size imageSize, std::vector<cv::Mat>& Ks);

    /**
     * @brief ����������
     * @param objectPoints ��������ϵ�еĽǵ�
     * @param imagePoints ͼ������ϵ�еĽǵ�
     * @param imageSize ͼ��ߴ�
     * @param cameraMatrix �ڲξ���
     * @param distCoeffs ���: �������
     * @param reprojectionError ���: ��ͶӰ���
     * @return �����Ƿ�ɹ�
     */
    bool computeDistortionCoefficients(const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2f>>& imagePoints,
        cv::Size imageSize, const cv::Mat& cameraMatrix,
        cv::Mat& distCoeffs, double& reprojectionError);

    /**
     * @brief ��ȡ�궨״̬��Ϣ
     * @return ״̬�����ַ���
     */
    std::string getStatus() const;

private:
    cv::Size m_boardSize;      // ���̸�ǵ�����
    float m_squareSize;        // ���̸񷽸�ʵ�ʳߴ�
    std::string m_status;      // �ڲ�״̬��Ϣ

    // �������̸�ǵ�
    bool findChessboardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners);

    // ������������ϵ�еĽǵ�
    void generateObjectPoints(std::vector<cv::Point3f>& objectPoints);

    // ��֤�궨���
    bool validateCalibration(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, double reprojectionError);

    // ����״̬��Ϣ
    void updateStatus(const std::string& message);
};