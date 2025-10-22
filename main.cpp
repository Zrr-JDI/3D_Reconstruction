#include <filesystem>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include"PCF.h"
#include"CloudGenerate.h"
using namespace std;
using namespace cv; 
namespace fs = std::filesystem;

bool Judge_Image(vector<bool>& image_judge)
{
    if (image_judge.size() == 0)
        return false;
    for (int i = 0; i < image_judge.size(); i++)
    {
        if (image_judge[i] == true)
            return true;
    }
    return false;
}

int main()
{

    // ���ձ�������
    vector<string> imageNames;                 // ÿ��ͼƬ�ļ���������ʵͼƬ·��һ�£�
    vector<cv::Mat> Rs;                               // ÿ���������ת���� (3x3)
    vector<cv::Mat> ts;                               // ÿ�������ƽ������ (3x1)
    vector<cv::Mat> Ks;                               // ÿ��������ڲξ��� (3x3)
    vector<cv::Point3d> points3D;                     // ȫ����ά�����꣨��������ϵ��
    vector<vector<cv::Point2d>> projections2D_all; // ÿ�������������ά��Ķ�άͶӰ����
    vector<vector<int>> viewIndices;             // ÿ��3D������Щ����б��۲⵽�����������
    vector<cv::Mat> images;                            // ��Ӧ��ԭʼͼ��������ȡ��ɫ��


    // ��Ҫ�õ����ݣ��������ձ���
    vector<vector<int>> point3DIds; // point3DIds[i][j] ��ʾ�� i ��ͼ�� j ���������Ӧ�� points3D ����
    vector<vector<cv::Point2d>>feature_points;              // ÿһ�ż�¼������
    cv::Mat diffcoeffs;                                    //����Ļ������
    cv::Mat K;                                              //����ڲ�(��������ڲ�һ��)
    int num = 0;                                              //��ӵĵڼ���ͼƬ
    vector<cv::Mat>Remaining_images;
    vector<string> Remaining_imageNames;

    // ָ��ͼƬ�ļ���·��
    string folder_path = "images";

    // ֧�ֵ�ͼƬ��չ��
    vector<string> image_extensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif" };



    // �����ļ���
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        string file_path = entry.path().string(); // ��ȡ����·��
        string extension = fs::path(file_path).extension().string();

        // ����չ��ת��ΪСд
        transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

        // ����Ƿ�Ϊ֧�ֵ�ͼƬ��ʽ
        if (find(image_extensions.begin(), image_extensions.end(), extension) != image_extensions.end()) {
            string relative_path = fs::relative(entry.path(), fs::current_path()).string();
            Remaining_imageNames.push_back(relative_path);
        }
    }


    // ��ȡ Ramining_imageNames �е�ͼƬ
    for (const auto& image_path : Remaining_imageNames) {
        // ʹ�� OpenCV ��ȡͼƬ
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR); // ��ȡΪ��ɫͼƬ
        if (!image.empty()) {
            Remaining_images.push_back(image);
            std::cout << "�ɹ���ȡͼƬ: " << image_path << std::endl;
        }
        else {
            std::cerr << "�޷���ȡͼƬ: " << image_path << std::endl;
            return -1;
        }
    }




    // ���У׼(��дK��diffcoeffs)















    // ��һ��������ȡ��ƥ��















    // ��ȹ��Ƴ������ɵ���




















    // ���ӵ�������
    
    vector<bool>image_judge;
    image_judge.resize(Remaining_images.size(), false);
    while (Judge_Image(image_judge))
    {
        image_judge.resize(image_judge.size(), true);
        for (int i = 0; i < image_judge.size(); i++)
        {
            int point_num=0;// ȡ�����
            int camera_number;
            vector<cv::Point2d> new_feature_points;// ��ȡ��ͼƬ��������
            vector<cv::Point2d> old_match_points;
            vector<cv::Point2d> new_match_points;
            // ����Ӧ��Remaining_Image[i]��������ƥ����ж�
            for (int j = 0; j < imageNames.size(); j++)
            {
                // ����������������point_num,����new_match_points��old_match_points,camera_number����������
                if ()
                {
                    camera_number = j;
                }
                else
                {

                }

            }

            if (point_num>=4)// ƥ��ɹ�
            {
                imageNames.push_back(Remaining_imageNames[i]);
                images.push_back(Remaining_images[i]);
                Remaining_imageNames.erase(Remaining_imageNames.begin() + i);
                Remaining_images.erase(Remaining_images.begin()+i);
                image_judge.erase(image_judge.begin() + i);
                feature_points.push_back(new_feature_points);
                point3DIds.push_back(vector<int>(new_feature_points.size(), -1));
                projections2D_all.push_back(new_match_points);
                num++;
                cv::Mat translate_vec;
                cv::Mat rotate_vec;
                point3DIds;

                // ƴ�� [R | t]
                Mat old_camera;
                hconcat(Rs[camera_number], ts[camera_number], old_camera);   // 3��4

                // ����ͶӰ����
                old_camera = Ks[camera_number] * old_camera;          // 3��4

                bool judge=IncreaseCloud(point3DIds,feature_points,camera_number,num, old_camera, old_match_points, points3D, new_match_points, K, diffcoeffs, translate_vec, rotate_vec, viewIndices);
                if (!judge) {
                    return -1;
                }
                Rs.push_back(rotate_vec);
                ts.push_back(translate_vec);
                break;
            }
            else// ƥ��ʧ��
            {
                image_judge[i] = false;
            }
        }
    }

    if (SimpleBundleAdjustAfterPnP(points3D, projections2D_all, Ks, Rs, ts) == false)
    {
        return -1;
    }

    Export_To_NVM(imageNames, Rs, ts, Ks, points3D, projections2D_all, viewIndices, images);

    //�����ؽ�����
    /*int ret = DensifyPointCloud();
    if (ret != 0)
    {
        cerr << "�����ؽ�ʧ�ܣ������ logs/DensifyPointCloud.log" << endl;
        return -1;
    }

    ret = ReconstructMesh();
    if (ret != 0)
    {
        cerr << "�����ؽ�ʧ�ܣ������ logs/ReconstructMesh.log" << endl;
        return -1;
    }

    ret = RefineMesh();
    if (ret != 0)
    {
        cerr << "�����Ż�ʧ�ܣ������ logs/RefineMesh.log" << endl;
        return -1;
    }

    ret = TextureMesh();
    if (ret != 0)
    {
        cerr << "������ͼʧ�ܣ������ logs/TextureMesh.log" << endl;
        return -1;
    }*/
	return 0;
}