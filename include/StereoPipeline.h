#pragma once
#ifndef STEREOPIPELINE_H
#define STEREOPIPELINE_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MultiViewStereo.h"
#include "SparseTriangulation.h"

// ��װ��������ͼִ�� E ���� -> �ָ� R,t -> ���ǻ� -> ����ϡ�����
// ����:
//   pts1, pts2 - ����ͼ���е�ƥ�����ص㣨��Ӧ˳��
//   K, distCoeffs - ����ڲ�����䣨distCoeffs Ŀǰδ����ȥ���䣬���������Ա���չ��
//   outPly - ��� PLY �ļ�������Ϊ���򲻱��棩
// ���:
//   R, t - ����ͼ1����ͼ2 �������ת��ƽ�ƣ�t ��λ������
//   points3D_out - ���ɵ�ϡ����ά�㣨���1 �ο�ϵ��
// ����:
//   true ��ʾ�������̳ɹ������ٲ���һ��3D�㣨�� outPly �ǿ���д�ļ��ɹ���
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