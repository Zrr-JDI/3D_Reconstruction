#include "SparseTriangulation.h"
#include <cmath>
#include <fstream>

bool TriangulateTwoViews(const cv::Mat& P1, const cv::Mat& P2,
                         const std::vector<cv::Point2d>& pts1,
                         const std::vector<cv::Point2d>& pts2,
                         std::vector<cv::Point3d>& points3D,
                         std::vector<unsigned char>& validMask,
                         double reprojThreshold)
{
    points3D.clear();
    validMask.clear();

    if (pts1.size() != pts2.size() || pts1.empty())
        return false;

    int N = static_cast<int>(pts1.size());

    // 构造 2xN Mat (double)
    cv::Mat pts1_2xN(2, N, CV_64F), pts2_2xN(2, N, CV_64F);
    for (int i = 0; i < N; ++i) {
        pts1_2xN.at<double>(0, i) = pts1[i].x;
        pts1_2xN.at<double>(1, i) = pts1[i].y;
        pts2_2xN.at<double>(0, i) = pts2[i].x;
        pts2_2xN.at<double>(1, i) = pts2[i].y;
    }

    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, pts1_2xN, pts2_2xN, points4D); // 4 x N
    points4D.convertTo(points4D, CV_64F);

    cv::Mat P1d, P2d;
    P1.convertTo(P1d, CV_64F);
    P2.convertTo(P2d, CV_64F);

    validMask.assign(N, 0);

    for (int i = 0; i < N; ++i) {
        double w = points4D.at<double>(3, i);
        if (std::fabs(w) < 1e-12) {
            validMask[i] = 0;
            continue;
        }
        double X = points4D.at<double>(0, i) / w;
        double Y = points4D.at<double>(1, i) / w;
        double Z = points4D.at<double>(2, i) / w;

        // 齐次点用于相机坐标变换
        cv::Mat Xh = (cv::Mat_<double>(4,1) << X, Y, Z, 1.0);
        cv::Mat x1h = P1d * Xh;
        cv::Mat x2h = P2d * Xh;

        double z1 = x1h.at<double>(2,0);
        double z2 = x2h.at<double>(2,0);

        // 深度必须为正
        if (z1 <= 0.0 || z2 <= 0.0) {
            validMask[i] = 0;
            continue;
        }

        double u1 = x1h.at<double>(0,0) / z1;
        double v1 = x1h.at<double>(1,0) / z1;
        double u2 = x2h.at<double>(0,0) / z2;
        double v2 = x2h.at<double>(1,0) / z2;

        double err1 = std::hypot(u1 - pts1[i].x, v1 - pts1[i].y);
        double err2 = std::hypot(u2 - pts2[i].x, v2 - pts2[i].y);

        if (err1 <= reprojThreshold && err2 <= reprojThreshold) {
            validMask[i] = 1;
            points3D.emplace_back(X, Y, Z);
        } else {
            validMask[i] = 0;
        }
    }

    return !points3D.empty();
}

bool SavePointCloudPLY(const std::string& filename,
                       const std::vector<cv::Point3d>& points3D)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open()) return false;

    ofs << "ply\nformat ascii 1.0\n";
    ofs << "element vertex " << points3D.size() << "\n";
    ofs << "property float x\nproperty float y\nproperty float z\n";
    ofs << "end_header\n";
    for (const auto& p : points3D) {
        ofs << p.x << " " << p.y << " " << p.z << "\n";
    }
    ofs.close();
    return true;
}