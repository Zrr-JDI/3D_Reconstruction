#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include <opencv2/opencv.hpp>
#include <vector>

class FeatureMatcher
{
public:
    // ���죺����ָ�� ORB �������� Lowe ratio
    FeatureMatcher(int nFeatures = 4000, float ratio = 0.75f);

    // ������ͼƬ��������⡢�����Ӽ�����ƥ�䣨Lowe Ratio Test��
    // ���룺img1, img2����ɫ��ҶȾ��ɣ�
    // �����kps1,kps2���ؼ��㣩��good_matches�����˺��ƥ�䣩��outImg�����ӻ�ͼ���� draw=false ��Ϊ�գ�
    // ���أ�ƥ��ɹ����� good_matches �ǿշ��� true�����򷵻� false
    bool matchImages(
        const cv::Mat& img1,
        const cv::Mat& img2,
        std::vector<cv::KeyPoint>& kps1,
        std::vector<cv::KeyPoint>& kps2,
        std::vector<cv::DMatch>& good_matches,
        cv::Mat& outImg,
        bool draw = true);

    // ���� ratio
    void setRatio(float r) { ratio_ = r; }

private:
    int nFeatures_;
    float ratio_;
    cv::Ptr<cv::ORB> orb_;
};

#endif // FEATURE_MATCHER_H
#pragma once
