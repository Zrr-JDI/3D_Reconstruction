#include "CameraCalibrator.h"


MonoCameraCalibrator::MonoCameraCalibrator(cv::Size boardSize, float squareSize)
    : m_boardSize(boardSize), m_squareSize(squareSize) {
    updateStatus("����궨����ʼ�����");
}

CalibrationResult MonoCameraCalibrator::calibrateCamera(const std::vector<std::string>& imagePaths) {
    CalibrationResult result;
    auto startTime = std::chrono::high_resolution_clock::now();

    updateStatus("��ʼ����궨����...");

    if (imagePaths.empty()) {
        result.errorMessage = "ͼ��·���б�Ϊ��";
        updateStatus("�궨ʧ��: " + result.errorMessage);
        return result;
    }

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Size imageSize(0, 0);
    int validImageCount = 0;

    updateStatus("������̸�ǵ�...");

    // 1. �������ͼ��Ľǵ�
    for (size_t i = 0; i < imagePaths.size(); i++) {
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            updateStatus("����: �޷���ȡͼ��: " + imagePaths[i]);
            continue;
        }

        // ��¼ͼ��ߴ磨ʹ�õ�һ����Чͼ��ĳߴ磩
        if (imageSize.width == 0) {
            imageSize = image.size();
        }

        std::vector<cv::Point2f> corners;
        if (findChessboardCorners(image, corners)) {
            // ���ɶ�Ӧ��3D��
            std::vector<cv::Point3f> objPoints;
            generateObjectPoints(objPoints);

            objectPoints.push_back(objPoints);
            imagePoints.push_back(corners);
            validImageCount++;

            updateStatus("�ɹ����ͼ�� " + std::to_string(i + 1) + " �Ľǵ�");
        }
        else {
            updateStatus("����: δ����ͼ�� " + std::to_string(i + 1) + " ���ҵ����̸�ǵ�");
        }
    }

    updateStatus("��Чͼ������: " + std::to_string(validImageCount) + "/" + std::to_string(imagePaths.size()));

    if (validImageCount < 5) {
        result.errorMessage = "��Чͼ���������㣬������Ҫ5����Ч�����̸�ͼ��";
        updateStatus("�궨ʧ��: " + result.errorMessage);
        return result;
    }

    if (imageSize.width == 0) {
        result.errorMessage = "�޷�ȷ��ͼ��ߴ�";
        updateStatus("�궨ʧ��: " + result.errorMessage);
        return result;
    }

    // 2. ��������ڲξ��� K
    updateStatus("��������ڲξ��� K...");
    std::vector<cv::Mat> Ks;
    if (!computeCameraIntrinsics(objectPoints, imagePoints, imageSize, Ks)) {
        result.errorMessage = "��������ڲξ���ʧ��";
        updateStatus("�궨ʧ��: " + result.errorMessage);
        return result;
    }

    // 3. ����������
    updateStatus("����������...");
    cv::Mat distCoeffs;
    double reprojectionError = 0.0;
    if (!computeDistortionCoefficients(objectPoints, imagePoints, imageSize, Ks[0],
        distCoeffs, reprojectionError)) {
        result.errorMessage = "����������ʧ��";
        updateStatus("�궨ʧ��: " + result.errorMessage);
        return result;
    }

    // 4. ��֤�궨���
    if (validateCalibration(Ks[0], distCoeffs, reprojectionError)) {
        result.success = true;
        result.Ks = Ks;
        result.distCoeffs = distCoeffs;
        result.reprojectionError = reprojectionError;

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        updateStatus("����궨�ɹ����! ��ͶӰ���: " +
            std::to_string(reprojectionError) + " ����, ��ʱ: " +
            std::to_string(duration.count()) + "ms");
    }
    else {
        result.errorMessage = "�궨�����֤ʧ��";
        updateStatus("�궨ʧ��: " + result.errorMessage);
    }

    return result;
}

bool MonoCameraCalibrator::computeCameraIntrinsics(const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    cv::Size imageSize, std::vector<cv::Mat>& Ks) {
    // ��ʼ���������
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

    // ʹ��OpenCV�ı궨���������ڲ�
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;

    double reprojectionError = cv::calibrateCamera(
        objectPoints, imagePoints, imageSize,
        cameraMatrix, distCoeffs,
        rvecs, tvecs,
        cv::CALIB_FIX_ASPECT_RATIO |
        cv::CALIB_ZERO_TANGENT_DIST |
        cv::CALIB_FIX_K3
    );

    // ����ڲξ������Ч��
    if (cameraMatrix.empty() || cameraMatrix.at<double>(0, 0) <= 0) {
        updateStatus("����: ������ڲξ�����Ч");
        return false;
    }

    // ���ڲξ�����ӵ����������
    Ks.clear();
    Ks.push_back(cameraMatrix);

    updateStatus("����ڲξ������ɹ�");
    updateStatus("���� fx: " + std::to_string(cameraMatrix.at<double>(0, 0)));
    updateStatus("���� fy: " + std::to_string(cameraMatrix.at<double>(1, 1)));
    updateStatus("���� cx: " + std::to_string(cameraMatrix.at<double>(0, 2)));
    updateStatus("���� cy: " + std::to_string(cameraMatrix.at<double>(1, 2)));

    return true;
}

bool MonoCameraCalibrator::computeDistortionCoefficients(const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    cv::Size imageSize, const cv::Mat& cameraMatrix,
    cv::Mat& distCoeffs, double& reprojectionError) {
    // ��ʼ������ϵ��
    distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

    // ʹ��OpenCV�ı궨��������������
    std::vector<cv::Mat> rvecs, tvecs;

    reprojectionError = cv::calibrateCamera(
        objectPoints, imagePoints, imageSize,
        const_cast<cv::Mat&>(cameraMatrix), distCoeffs,
        rvecs, tvecs,
        cv::CALIB_USE_INTRINSIC_GUESS |
        cv::CALIB_FIX_ASPECT_RATIO |
        cv::CALIB_ZERO_TANGENT_DIST |
        cv::CALIB_FIX_K3
    );

    // �������������Ч��
    if (distCoeffs.empty()) {
        updateStatus("����: ����Ļ��������Ч");
        return false;
    }

    updateStatus("�����������ɹ�");
    updateStatus("������� k1: " + std::to_string(distCoeffs.at<double>(0, 0)));
    updateStatus("������� k2: " + std::to_string(distCoeffs.at<double>(0, 1)));
    updateStatus("������� p1: " + std::to_string(distCoeffs.at<double>(0, 2)));
    updateStatus("������� p2: " + std::to_string(distCoeffs.at<double>(0, 3)));
    updateStatus("������� k3: " + std::to_string(distCoeffs.at<double>(0, 4)));
    updateStatus("��ͶӰ���: " + std::to_string(reprojectionError) + " ����");

    return true;
}

bool MonoCameraCalibrator::findChessboardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners) {
    bool found = cv::findChessboardCorners(image, m_boardSize, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

    if (found) {
        // �����ؼ��ǵ㾫ȷ��
        cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
    }

    return found;
}

void MonoCameraCalibrator::generateObjectPoints(std::vector<cv::Point3f>& objectPoints) {
    objectPoints.clear();
    for (int i = 0; i < m_boardSize.height; i++) {
        for (int j = 0; j < m_boardSize.width; j++) {
            objectPoints.push_back(cv::Point3f(j * m_squareSize, i * m_squareSize, 0));
        }
    }
}

bool MonoCameraCalibrator::validateCalibration(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, double reprojectionError) {
    bool valid = !cameraMatrix.empty() &&
        !distCoeffs.empty() &&
        reprojectionError < 2.0 && // ��ͶӰ���С��2����
        cameraMatrix.at<double>(0, 0) > 0; // ������Ч

    if (!valid) {
        updateStatus("�궨�����֤ʧ��: ��ͶӰ���=" +
            std::to_string(reprojectionError));
    }

    return valid;
}

std::string MonoCameraCalibrator::getStatus() const {
    return m_status;
}

void MonoCameraCalibrator::updateStatus(const std::string& message) {
    m_status = message;
    std::cout << "[CameraCalibrator] " << message << std::endl;
}