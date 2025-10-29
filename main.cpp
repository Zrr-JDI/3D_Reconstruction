#include <filesystem>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include"PCF.h"
#include"CloudGenerate.h"
#include"FeatureDetection.h"
#include"CameraCalibrator.h"
#include"StereoPipeline.h"
#include "MultiViewStereo.h"
#include "SparseTriangulation.h"
using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// 将 KeyPoint 向量转为 Point2d 向量
std::vector<cv::Point2d> KeyPointsToPoints2D(const std::vector<cv::KeyPoint>& keypoints)
{
    std::vector<cv::Point2d> points;
    points.reserve(keypoints.size());
    for (const auto& kp : keypoints)
        points.emplace_back(kp.pt.x, kp.pt.y);
    return points;
}

// 保存标定参数到YAML文件
bool saveCalibrationParameters(const std::string& filename,
    const std::vector<cv::Mat>& Ks,
    const cv::Mat& distCoeffs,
    double reprojectionError,
    cv::Size boardSize,
    float squareSize) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    if (!fs.isOpened()) {
        std::cerr << "错误: 无法创建文件 " << filename << std::endl;
        return false;
    }

    // 保存标定参数
    fs << "calibration_date" << "相机标定结果";

    // 内参矩阵 K
    if (!Ks.empty()) {
        fs << "camera_matrix" << Ks[0];
    }

    // 畸变参数
    fs << "distortion_coefficients" << distCoeffs;

    // 标定精度信息
    fs << "reprojection_error" << reprojectionError;

    // 棋盘格信息
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    fs.release();

    std::cout << "相机参数已保存到: " << filename << std::endl;
    return true;
}

// 打印相机参数
void printCameraParameters(const std::vector<cv::Mat>& Ks, const cv::Mat& distCoeffs, double reprojectionError) {
    std::cout << "\n=== 相机标定结果 ===" << std::endl;
    std::cout << "重投影误差: " << reprojectionError << " 像素" << std::endl;

    if (!Ks.empty()) {
        std::cout << "\n内参矩阵 K:" << std::endl;
        std::cout << "[" << Ks[0].at<double>(0, 0) << ", "
            << Ks[0].at<double>(0, 1) << ", "
            << Ks[0].at<double>(0, 2) << "]" << std::endl;
        std::cout << "[" << Ks[0].at<double>(1, 0) << ", "
            << Ks[0].at<double>(1, 1) << ", "
            << Ks[0].at<double>(1, 2) << "]" << std::endl;
        std::cout << "[" << Ks[0].at<double>(2, 0) << ", "
            << Ks[0].at<double>(2, 1) << ", "
            << Ks[0].at<double>(2, 2) << "]" << std::endl;
    }

    std::cout << "\n畸变参数: [";
    if (!distCoeffs.empty()) {
        for (int i = 0; i < distCoeffs.cols; i++) {
            std::cout << distCoeffs.at<double>(0, i);
            if (i < distCoeffs.cols - 1) std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
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
    vector<vector<cv::KeyPoint>>Feature_points;              //记录特征点具体数据用于匹配
    vector<vector<cv::Point2d>>feature_points;              // 每一张记录特征点坐标数据
    vector<cv::Mat>descriptors;                             //保存特征点的描述
    cv::Mat diffcoeffs;                                    //相机的畸变参数
    cv::Mat K;                                              //相机内参(所有相机内参一样)
    int num = 0;                                              //添加的第几张图片
    vector<cv::Mat>Remaining_images;
    vector<string> Remaining_imageNames;

    // 指定图片文件夹路径
    string folder_path = "images";
    string output_path = "camera";

    // 支持的图片扩展名
    vector<string> image_extensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tiff"};

    fs::path absolute_path = fs::absolute(folder_path);

    // 打印绝对路径（注意：可能包含平台特定的路径分隔符）
    std::cout << "Absolute path: " << absolute_path << std::endl;

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


    Size boardSize(9, 6);      // 棋盘格内角点数量 (width, height)
    float squareSize = 0.025f;     // 棋盘格方格实际尺寸(米)

    if (Remaining_imageNames.empty()) {
        std::cerr << "错误: 在目录 " << folder_path << " 中未找到图像文件" << std::endl;
        std::cerr << "支持的格式: .jpg, .jpeg, .png, .bmp, .tiff" << std::endl;
        std::cerr << "请将棋盘格图片放入 " << folder_path << " 目录后重新运行程序" << std::endl;
        return -1;
    }


    // 相机校准(填写K，diffcoeffs)
    MonoCameraCalibrator calibrator(boardSize, squareSize);

    std::cout << "\n开始执行相机标定..." << std::endl;
    std::cout << "棋盘格尺寸: " << boardSize.width << "x" << boardSize.height << " 角点" << std::endl;
    std::cout << "方格尺寸: " << squareSize * 1000 << " mm" << std::endl;
    std::cout << "图像数量: " << Remaining_imageNames.size() << " 张" << std::endl;

    // 执行相机标定
    CalibrationResult result = calibrator.calibrateCamera(Remaining_imageNames);

    if (result.success) {
        std::cout << "\n✓ 相机标定成功!" << std::endl;

        // 打印相机参数
        printCameraParameters(result.Ks, result.distCoeffs, result.reprojectionError);

        // 保存参数到YAML文件
        std::string yamlFile = output_path + "\\camera.yml";
        if (saveCalibrationParameters(yamlFile, result.Ks, result.distCoeffs, result.reprojectionError, boardSize, squareSize)) {
            std::cout << "\n✓ 相机参数已保存到: " << yamlFile << std::endl;

            // 显示YAML文件内容预览
            std::cout << "\nYAML文件内容预览:" << std::endl;
            std::ifstream file(yamlFile.c_str());
            if (file.is_open()) {
                std::string line;
                int count = 0;
                while (std::getline(file, line) && count < 10) {
                    std::cout << line << std::endl;
                    count++;
                }
                file.close();
            }
            else {
                std::cerr << "警告: 无法打开YAML文件进行预览" << std::endl;
            }
        }
        else {
            std::cerr << "错误: 保存YAML文件失败" << std::endl;
        }

        std::cout << "\n标定完成! 所有结果已保存到 " << output_path << " 目录" << std::endl;

    }
    else {
        std::cerr << "\n✗ 相机标定失败: " << result.errorMessage << std::endl;
        std::cerr << "当前状态: " << calibrator.getStatus() << std::endl;
        return -1;
    }

    diffcoeffs = result.distCoeffs;
    K = result.Ks[0];



    // 第一轮特征提取和匹配
    if (Remaining_imageNames.size() < 2)
    {
        cerr << "照片数量不足" << endl;
        return -1;
    }
    FeatureMatcher matcher;
    vector<cv::KeyPoint> key_points_0,key_points_1;
    cv::Mat des_0,des_1;
    matcher.extractFeatures(Remaining_images[0], key_points_0,des_0);
    matcher.extractFeatures(Remaining_images[1], key_points_1, des_1);
    vector<cv::Point2d> match_points_0;
    vector<cv::Point2d> match_points_1;
    matcher.matchFeatures(key_points_0,des_0,key_points_1,des_1,match_points_0,match_points_1);
    Feature_points.push_back(key_points_0);
    Feature_points.push_back(key_points_1);
    feature_points.push_back(KeyPointsToPoints2D(key_points_0));
    feature_points.push_back(KeyPointsToPoints2D(key_points_1));
    imageNames.push_back(Remaining_imageNames[0]);
    imageNames.push_back(Remaining_imageNames[1]);
    images.push_back(Remaining_images[0]);
    images.push_back(Remaining_images[1]);
    Remaining_imageNames.erase(Remaining_imageNames.begin());
    Remaining_imageNames.erase(Remaining_imageNames.begin());
    Remaining_images.erase(Remaining_images.begin());
    Remaining_images.erase(Remaining_images.begin());
    projections2D_all.push_back(match_points_0);
    projections2D_all.push_back(match_points_1);
    descriptors.push_back(des_0);
    descriptors.push_back(des_1);
    Ks.push_back(result.Ks[0]);
    Ks.push_back(result.Ks[1]);
    result.Ks.erase(result.Ks.begin());
    result.Ks.erase(result.Ks.begin());
    num += 2;





    // 深度估计初步生成点云

    Mat R;
    Mat t;
    vector<Point3d> points3D_out;
    if (!RunTwoViewReconstruction(match_points_0, match_points_1, Ks[0], diffcoeffs, "scene.ply", R, t, points3D_out, points3D, projections2D_all, point3DIds, 1.0, 2.0))
    {
        cerr << "深度估计错误" << endl;
        return -1;
    }
    Rs.push_back(cv::Mat::eye(3, 3, CV_64F)); 
    ts.push_back(cv::Mat::zeros(3, 1, CV_64F));
    Rs.push_back(R);
    ts.push_back(t);




    bool judge;
    // 增加点云数量
    do
    {
        judge = false;
        for (int i = 0; i < Remaining_imageNames.size(); i++)
        {
            int point_num=0;// 取最大数
            int camera_number;
            vector<cv::KeyPoint> new_key_points;
            cv::Mat des;
            matcher.extractFeatures(Remaining_images[i], new_key_points,des);
            vector<cv::Point2d> new_feature_points;// 获取该图片的特征点
            vector<cv::Point2d> old_match_points;
            vector<cv::Point2d> new_match_points;
            // 对相应的Remaining_Image[i]进行特征匹配和判定
            for (int j = 0; j < imageNames.size(); j++)
            {
                vector<cv::Point2d> match_pointsA;
                vector<cv::Point2d> match_pointsB;
                matcher.matchFeatures(Feature_points[j],descriptors[j],new_key_points,des,match_pointsA,match_pointsB);
                // 如果特征点个数大于point_num,更新new_match_points和old_match_points,camera_number否则不做处理
                if (match_pointsA.size()>new_match_points.size())
                {
                    point_num = match_pointsA.size();
                    camera_number = j;
                    new_match_points = match_pointsA;
                    old_match_points = match_pointsB;
                }
            }

            if (point_num>=4)// 匹配成功
            {
                imageNames.push_back(Remaining_imageNames[i]);
                images.push_back(Remaining_images[i]);
                Remaining_imageNames.erase(Remaining_imageNames.begin() + i);
                Remaining_images.erase(Remaining_images.begin()+i);
                Feature_points.push_back(new_key_points);
                descriptors.push_back(des);
                new_feature_points = KeyPointsToPoints2D(new_key_points);
                feature_points.push_back(new_feature_points);
                point3DIds.push_back(vector<int>(new_feature_points.size(), -1));
                projections2D_all.push_back(new_match_points);
                num++;
                cv::Mat translate_vec;
                cv::Mat rotate_vec;
                point3DIds;
                judge = true;
                // 拼接 [R | t]
                Mat old_camera;
                hconcat(Rs[camera_number], ts[camera_number], old_camera);   // 3×4

                // 计算投影矩阵
                old_camera = Ks[camera_number] * old_camera;          // 3×4

                bool success=IncreaseCloud(point3DIds,feature_points,camera_number,num, old_camera, old_match_points, points3D, new_match_points, K, diffcoeffs, translate_vec, rotate_vec, viewIndices);
                if (!success) {
                    return -1;
                }
                Rs.push_back(rotate_vec);
                ts.push_back(translate_vec);
                Ks.push_back(result.Ks[i]);
                result.Ks.erase(result.Ks.begin()+i);
                break;
            }
        }
    } while (judge == true && Remaining_imageNames.size() != 0);

    if (SimpleBundleAdjustAfterPnP(points3D, projections2D_all, Ks, Rs, ts) == false)
    {
        return -1;
    }
    

    Export_To_NVM(imageNames, Rs, ts, Ks, points3D, projections2D_all, viewIndices, images);

    string sourceImagesDir = "images"; 
    string targetMvsDir = "MVS";       
    string targetImagesDir = targetMvsDir + "/images"; 

    if (!fs::exists(targetMvsDir)) 
    {
        std::cerr << "错误：MVS 文件夹不存在 - " << targetMvsDir << std::endl;
        return -1;
    }

    if (!fs::exists(sourceImagesDir)) 
    {
        std::cerr << "错误：images 文件夹不存在 - " << sourceImagesDir << std::endl;
        return -1;
    }

    if (fs::exists(targetImagesDir)) 
    {
        std::cout << "目标路径已存在 images 文件夹，将覆盖 - " << targetImagesDir << std::endl;
        fs::remove_all(targetImagesDir); 
    }

    fs::copy(sourceImagesDir, targetImagesDir, fs::copy_options::recursive);

    if (!EnsureLogDirectory())
    {
        cout << "创建logs日志目录失败" << endl;
        return -1;
    }

    //表面重建部分
    int ret = InterfaceVisualSFM();
    if (ret != 0)
    {
        cerr << "生成.mvs失败，详情见 logs/InterfaceVisualSFM.log" << endl;
        return -1;
    }

    ret = DensifyPointCloud();
    if (ret != 0)
    {
        cerr << "稠密重建失败，详情见 logs/DensifyPointCloud.log" << endl;
        return -1;
    }

    ret = ReconstructMesh();
    if (ret != 0)
    {
        cerr << "曲面重建失败，详情见 logs/ReconstructMesh.log" << endl;
        return -1;
    }

    ret = RefineMesh();
    if (ret != 0)
    {
        cerr << "网格优化失败，详情见 logs/RefineMesh.log" << endl;
        return -1;
    }

    ret = TextureMesh();
    if (ret != 0)
    {
        cerr << "纹理贴图失败，详情见 logs/TextureMesh.log" << endl;
        return -1;
    }

	return 0;
}