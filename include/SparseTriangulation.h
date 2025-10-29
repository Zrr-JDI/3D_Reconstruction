#pragma once
#ifndef SPARSETRIANGULATION_H
#define SPARSETRIANGULATION_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// ��������ͼ�����ǻ���������� + ��ͶӰ���ɸѡ
// P1, P2: 3x4 ͶӰ�����Ѱ����ڲ� K��
// pts1, pts2: ��Ӧ����������������ͬ���ȣ�
// points3D: ���ͨ��ɸѡ����ά�㣨���1�ο�ϵ��
// validMask: ����������ȳ��� mask��1: ��Ч��0: ��Ч��
// reprojThreshold: ��ͶӰ�����ֵ�����أ�
// ���� true ��ʾ������һ����Ч��
bool TriangulateTwoViews(const cv::Mat& P1, const cv::Mat& P2,
                         const std::vector<cv::Point2d>& pts1,
                         const std::vector<cv::Point2d>& pts2,
                         std::vector<cv::Point3d>& points3D,
                         std::vector<unsigned char>& validMask,
                         double reprojThreshold = 3.0);

// ��ϡ�����д�� ASCII PLY�������㣩
// ���� true ��ʾд��ɹ�
bool SavePointCloudPLY(const std::string& filename,
                       const std::vector<cv::Point3d>& points3D);

#endif // SPARSETRIANGULATION_H