#pragma once
#ifndef MULTIVIEWSTEREO_H
#define MULTIVIEWSTEREO_H

#include <vector>
#include <opencv2/opencv.hpp>

// ���㱾�ʾ���ʹ�� RANSAC��
// pts1, pts2: �������꣨ͬһ��������ϵ��
// K: ����ڲ�
// E: ������ʾ���
// inlierMask: ����ڵ����� (CV_8U)
// ransacThresh: RANSAC ��ͶӰ��ֵ�����أ�
bool ComputeEssentialMatrix(const std::vector<cv::Point2d>& pts1,
                            const std::vector<cv::Point2d>& pts2,
                            const cv::Mat& K,
                            cv::Mat& E,
                            cv::Mat& inlierMask,
                            double ransacThresh = 1.0);

// �ӱ��ʾ���ָ������̬ R,t
// E: ���ʾ���
// pts1, pts2: ��Ӧ���ص㣨���� recoverPose �����룩
// K: ����ڲ�
// R, t: ��������ת��ƽ�ƣ�t Ϊ��λ��������
// inlierMask: ��ѡ��recoverPose ������ڵ����루CV_8U��
bool RecoverPoseFromEssential(const cv::Mat& E,
                              const std::vector<cv::Point2d>& pts1,
                              const std::vector<cv::Point2d>& pts2,
                              const cv::Mat& K,
                              cv::Mat& R,
                              cv::Mat& t,
                              cv::Mat& inlierMask);

#endif // MULTIVIEWSTEREO_H