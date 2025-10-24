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

    // 最终保存数据
    vector<string> imageNames;                 // 每张图片文件名（与真实图片路径一致）
    vector<Mat> Rs;                               // 每张相机的旋转矩阵 (3x3)
    vector<Mat> ts;                               // 每张相机的平移向量 (3x1)
    vector<Mat> Ks;                               // 每张相机的内参矩阵 (3x3)
    vector<Point3d> points3D;                     // 全局三维点坐标（世界坐标系）
    vector<vector<Point2d>> projections2D_all; // 每个相机上所有三维点的二维投影坐标
    vector<vector<int>> viewIndices;             // 每个3D点在哪些相机中被观测到（相机索引）
    vector<Mat> images;                            // 对应的原始图像（用于提取颜色）


    // 需要用到数据，不用最终保存
    vector<vector<int>> point3DIds; // point3DIds[i][j] 表示第 i 张图第 j 个特征点对应的 points3D 索引
    vector<vector<cv::Point2d>>feature_points;              // 每一张记录特征点
    cv::Mat diffcoeffs;                                    //相机的畸变参数
    cv::Mat K;                                              //相机内参(所有相机内参一样)
    int num = 0;                                              //添加的第几张图片
    vector<cv::Mat>Remaining_images;
    vector<string> Remaining_imageNames;

    // 指定图片文件夹路径
    string folder_path = "images";

    // 支持的图片扩展名
    vector<string> image_extensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif" };



    // 遍历文件夹
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        string file_path = entry.path().string(); // 获取完整路径
        string extension = fs::path(file_path).extension().string();

        // 将扩展名转换为小写
        transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

        // 检查是否为支持的图片格式
        if (find(image_extensions.begin(), image_extensions.end(), extension) != image_extensions.end()) {
            string relative_path = fs::relative(entry.path(), fs::current_path()).string();
            Remaining_imageNames.push_back(relative_path);
        }
    }


    // 读取 Ramining_imageNames 中的图片
    for (const auto& image_path : Remaining_imageNames) {
        // 使用 OpenCV 读取图片
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR); // 读取为彩色图片
        if (!image.empty()) {
            Remaining_images.push_back(image);
            std::cout << "成功读取图片: " << image_path << std::endl;
        }
        else {
            std::cerr << "无法读取图片: " << image_path << std::endl;
            return -1;
        }
    }




    // 相机校准(填写K，diffcoeffs)















    // 第一轮特征提取和匹配















    // 深度估计初步生成点云




















    // 增加点云数量
    
    vector<bool>image_judge;
    image_judge.resize(Remaining_images.size(), false);
    while (Judge_Image(image_judge))
    {
        image_judge.resize(image_judge.size(), true);
        for (int i = 0; i < image_judge.size(); i++)
        {
            int point_num=0;// 取最大数
            int camera_number;
            vector<cv::Point2d> new_feature_points;// 获取该图片的特征点
            vector<cv::Point2d> old_match_points;
            vector<cv::Point2d> new_match_points;
            // 对相应的Remaining_Image[i]进行特征匹配和判定
            for (int j = 0; j < imageNames.size(); j++)
            {
                // 如果特征点个数大于point_num,更新new_match_points和old_match_points,camera_number否则不做处理
                if ()
                {
                    camera_number = j;
                }
                else
                {

                }

            }

            if (point_num>=4)// 匹配成功
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

                // 拼接 [R | t]
                Mat old_camera;
                hconcat(Rs[camera_number], ts[camera_number], old_camera);   // 3×4

                // 计算投影矩阵
                old_camera = Ks[camera_number] * old_camera;          // 3×4

                bool judge=IncreaseCloud(point3DIds,feature_points,camera_number,num, old_camera, old_match_points, points3D, new_match_points, K, diffcoeffs, translate_vec, rotate_vec, viewIndices);
                if (!judge) {
                    return -1;
                }
                Rs.push_back(rotate_vec);
                ts.push_back(translate_vec);
                break;
            }
            else// 匹配失败
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

    //string sourceImagesDir = "images"; 
    //string targetMvsDir = "MVS";       
    //string targetImagesDir = targetMvsDir + "/images"; 

    //if (!fs::exists(targetMvsDir)) 
    //{
    //    std::cerr << "错误：MVS 文件夹不存在 - " << targetMvsDir << std::endl;
    //    return -1;
    //}

    //if (!fs::exists(sourceImagesDir)) 
    //{
    //    std::cerr << "错误：images 文件夹不存在 - " << sourceImagesDir << std::endl;
    //    return -1;
    //}

    //if (fs::exists(targetImagesDir)) 
    //{
    //    std::cout << "目标路径已存在 images 文件夹，将覆盖 - " << targetImagesDir << std::endl;
    //    fs::remove_all(targetImagesDir); 
    //}

    //fs::copy(sourceImagesDir, targetImagesDir, fs::copy_options::recursive);

    //if (!EnsureLogDirectory())
    //{
    //    cout << "创建logs日志目录失败" << endl;
    //    return -1;
    //}

    ////表面重建部分
    //int ret = InterfaceVisualSFM();
    //if (ret != 0)
    //{
    //    cerr << "生成.mvs失败，详情见 logs/InterfaceVisualSFM.log" << endl;
    //    return -1;
    //}

    //ret = DensifyPointCloud();
    //if (ret != 0)
    //{
    //    cerr << "稠密重建失败，详情见 logs/DensifyPointCloud.log" << endl;
    //    return -1;
    //}

    //ret = ReconstructMesh();
    //if (ret != 0)
    //{
    //    cerr << "曲面重建失败，详情见 logs/ReconstructMesh.log" << endl;
    //    return -1;
    //}

    //ret = RefineMesh();
    //if (ret != 0)
    //{
    //    cerr << "网格优化失败，详情见 logs/RefineMesh.log" << endl;
    //    return -1;
    //}

    //ret = TextureMesh();
    //if (ret != 0)
    //{
    //    cerr << "纹理贴图失败，详情见 logs/TextureMesh.log" << endl;
    //    return -1;
    //}

	return 0;
}