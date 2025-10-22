#include"CloudGenerate.h"


bool PnP(std::vector<cv::Point3d>& model_points, std::vector<cv::Point2d>& image_points, cv::Mat& camera_matrix, cv::Mat& diffcoeffs, cv::Mat& translate_vec, cv::Mat& rotate_vec)
{
	bool judge;
	judge=cv::solvePnP(model_points, image_points, camera_matrix, diffcoeffs, translate_vec, rotate_vec, false, cv::SOLVEPNP_ITERATIVE);
	if (judge == false)
	{
		std::cout<<"PnP处理失败" << std::endl;
		return false;
	}
	return true;

}

bool IncreaseCloud(std::vector<std::vector<int>>& point3DIds,std::vector<std::vector<cv::Point2d>>& feature_points, int camera_number,int num, cv::Mat& old_camera, std::vector<cv::Point2d>& old_image_points, std::vector<cv::Point3d>& model_points, std::vector<cv::Point2d>& image_points, cv::Mat& camera_matrix, cv::Mat& diffcoeffs, cv::Mat& translate_vec, cv::Mat& rotate_vec, std::vector<std::vector<int>>& viewIndices)
{
    bool judge;
    judge=PnP(model_points, image_points, camera_matrix, diffcoeffs, translate_vec, rotate_vec);
    if (judge == false)
        return false;


    // Rodrigues 转换为旋转矩阵
    cv::Mat R;
    cv::Rodrigues(rotate_vec, R);

    // 拼接成 [R|t]
    cv::Mat Rt;
    cv::hconcat(R, translate_vec, Rt); 

    // 构造新相机投影矩阵
    cv::Mat new_camera = camera_matrix * Rt;

    // 三角化
    cv::Mat points4D;
    cv::triangulatePoints(old_camera, new_camera, old_image_points, image_points, points4D);

    if (points4D.empty() || points4D.cols == 0)
    {
        std::cout << "三角化失败" << std::endl;
        return false;
    }


    // 转换为非齐次坐标
    std::vector<cv::Point3d> new_points;
    for (int i = 0; i < points4D.cols; i++)
    {
        cv::Mat x = points4D.col(i);
        x /= x.at<float>(3); 
        new_points.emplace_back(
            x.at<float>(0),
            x.at<float>(1),
            x.at<float>(2)
        );
    }

    for (int i = 0; i < old_image_points.size(); i++)
    {
        // 取出匹配点
        cv::Point2d p_old = old_image_points[i];
        cv::Point2d p_new = image_points[i];

        // 在旧相机特征点中查找
        auto it_old = std::find_if(
            feature_points[camera_number].begin(),
            feature_points[camera_number].end(),
            [&](const cv::Point2d& p)
            {
                return std::fabs(p.x - p_old.x) < 1e-6 && std::fabs(p.y - p_old.y) < 1e-6;
            });

        // 在新相机特征点中查找
        auto it_new = std::find_if(
            feature_points[num].begin(),
            feature_points[num].end(),
            [&](const cv::Point2d& p)
            {
                return std::fabs(p.x - p_new.x) < 1e-6 && std::fabs(p.y - p_new.y) < 1e-6;
            });

        if (it_old == feature_points[camera_number].end() ||
            it_new == feature_points[num].end())
        {
            // 没找到匹配索引
            continue;
        }

        int old_idx = std::distance(feature_points[camera_number].begin(), it_old);
        int new_idx = std::distance(feature_points[num].begin(), it_new);

        int pid = point3DIds[camera_number][old_idx];

        if (pid >= 0)
        {
            point3DIds[num][new_idx] = pid;
            viewIndices[pid].push_back(num);
        }
        else
        {
            int new_pid = model_points.size();
            model_points.push_back(new_points[i]);
            point3DIds[camera_number][old_idx] = new_pid;
            point3DIds[num][new_idx] = new_pid;
            viewIndices.push_back({ camera_number, num });
        }
    }

}


// 简化版 Bundle Adjustment
bool SimpleBundleAdjustAfterPnP(
    std::vector<cv::Point3d>& points3D,
    const std::vector<std::vector<cv::Point2d>>& projections2D_all,
    const std::vector<cv::Mat>& Ks,
    std::vector<cv::Mat>& Rs,
    std::vector<cv::Mat>& ts,
    int iterations = 5
)
{
    int numCams = Rs.size();
    int numPoints = points3D.size();
    bool updated = false; // 是否至少更新过一次

    // -------- 只更新位姿，不更新点 --------
    for (int iter = 0; iter < iterations; ++iter) {
        for (int camIdx = 0; camIdx < numCams; ++camIdx) {
            std::vector<cv::Point3d> visiblePoints;
            std::vector<cv::Point2d> imagePoints;

            for (int ptIdx = 0; ptIdx < numPoints; ++ptIdx) {
                if (ptIdx >= projections2D_all[camIdx].size()) continue;
                cv::Point2d uv = projections2D_all[camIdx][ptIdx];
                if (uv.x >= 0 && uv.y >= 0) {
                    visiblePoints.push_back(points3D[ptIdx]);
                    imagePoints.push_back(uv);
                }
            }

            if (visiblePoints.size() < 4) continue;

            cv::Mat rvec, tvec;
            cv::Rodrigues(Rs[camIdx], rvec);
            tvec = ts[camIdx];

            bool ok = cv::solvePnP(
                visiblePoints,
                imagePoints,
                Ks[camIdx],
                cv::Mat(),
                rvec,
                tvec,
                true,
                cv::SOLVEPNP_ITERATIVE
            );

            if (!ok) continue;

            cv::Rodrigues(rvec, Rs[camIdx]);
            ts[camIdx] = tvec;
            updated = true;
        }
    }

    if (!updated) return false; // 如果没有相机更新，返回false

    // -------- 最后统一更新点云 --------
    for (int ptIdx = 0; ptIdx < numPoints; ++ptIdx) {
        cv::Point3d& P = points3D[ptIdx];
        cv::Point3d delta(0, 0, 0);
        int count = 0;

        for (int camIdx = 0; camIdx < numCams; ++camIdx) {
            if (ptIdx >= projections2D_all[camIdx].size()) continue;
            cv::Point2d uv = projections2D_all[camIdx][ptIdx];
            if (uv.x < 0 || uv.y < 0) continue;

            cv::Mat R = Rs[camIdx];
            cv::Mat t = ts[camIdx];
            cv::Mat X = (cv::Mat_<double>(3, 1) << P.x, P.y, P.z);
            cv::Mat proj = Ks[camIdx] * (R * X + t);
            proj /= proj.at<double>(2);

            cv::Point2d error(uv.x - proj.at<double>(0), uv.y - proj.at<double>(1));
            delta.x += 0.001 * error.x;
            delta.y += 0.001 * error.y;
            delta.z += 0.001 * ((uv.x + uv.y) / 2 - proj.at<double>(0) / 2); // 简单微调 z
            count++;
        }

        if (count > 0) {
            P.x += delta.x / count;
            P.y += delta.y / count;
            P.z += delta.z / count;
        }
    }

    return true; // 成功执行
}



// 辅助函数：旋转矩阵 -> 四元数 (qw, qx, qy, qz)
cv::Vec4d rotationMatrixToQuaternion(const cv::Mat& R)
{
    CV_Assert(R.rows == 3 && R.cols == 3);
    cv::Vec4d q;
    double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
    if (trace > 0) {
        double s = 0.5 / sqrt(trace + 1.0);
        q[0] = 0.25 / s;
        q[1] = (R.at<double>(2, 1) - R.at<double>(1, 2)) * s;
        q[2] = (R.at<double>(0, 2) - R.at<double>(2, 0)) * s;
        q[3] = (R.at<double>(1, 0) - R.at<double>(0, 1)) * s;
    }
    else {
        if (R.at<double>(0, 0) > R.at<double>(1, 1) && R.at<double>(0, 0) > R.at<double>(2, 2)) {
            double s = 2.0 * sqrt(1.0 + R.at<double>(0, 0) - R.at<double>(1, 1) - R.at<double>(2, 2));
            q[0] = (R.at<double>(2, 1) - R.at<double>(1, 2)) / s;
            q[1] = 0.25 * s;
            q[2] = (R.at<double>(0, 1) + R.at<double>(1, 0)) / s;
            q[3] = (R.at<double>(0, 2) + R.at<double>(2, 0)) / s;
        }
        else if (R.at<double>(1, 1) > R.at<double>(2, 2)) {
            double s = 2.0 * sqrt(1.0 + R.at<double>(1, 1) - R.at<double>(0, 0) - R.at<double>(2, 2));
            q[0] = (R.at<double>(0, 2) - R.at<double>(2, 0)) / s;
            q[1] = (R.at<double>(0, 1) + R.at<double>(1, 0)) / s;
            q[2] = 0.25 * s;
            q[3] = (R.at<double>(1, 2) + R.at<double>(2, 1)) / s;
        }
        else {
            double s = 2.0 * sqrt(1.0 + R.at<double>(2, 2) - R.at<double>(0, 0) - R.at<double>(1, 1));
            q[0] = (R.at<double>(1, 0) - R.at<double>(0, 1)) / s;
            q[1] = (R.at<double>(0, 2) + R.at<double>(2, 0)) / s;
            q[2] = (R.at<double>(1, 2) + R.at<double>(2, 1)) / s;
            q[3] = 0.25 * s;
        }
    }
    return q;
}

// 导出为 VisualSFM 格式的 NVM 文件
void Export_To_NVM(
    const std::vector<std::string>& imageNames,                   // 每张图片文件名（与真实图片路径一致）
    const std::vector<cv::Mat>& Rs,                               // 每张相机的旋转矩阵 (3x3)
    const std::vector<cv::Mat>& ts,                               // 每张相机的平移向量 (3x1)
    const std::vector<cv::Mat>& Ks,                               // 每张相机的内参矩阵 (3x3)
    const std::vector<cv::Point3d>& points3D,                     // 全局三维点坐标（世界坐标系）
    const std::vector<std::vector<cv::Point2d>>& projections2D_all, // 每个相机上所有三维点的二维投影坐标
    const std::vector<std::vector<int>>& viewIndices,             // 每个3D点在哪些相机中被观测到（相机索引）
    const std::vector<cv::Mat>& images                            // 对应的原始图像（用于提取颜色）
) {
    std::ofstream file("scene.nvm");
    if (!file.is_open()) {
        std::cerr << "无法打开输出文件 scene.nvm" << std::endl;
        return;
    }

    file << "NVM_V3\n\n";

    // 写入相机信息
    file << imageNames.size() << "\n";

    for (size_t i = 0; i < imageNames.size(); ++i) {
        double f = Ks[i].at<double>(0, 0); // 焦距（fx）
        const cv::Mat& R = Rs[i];
        const cv::Mat& t = ts[i];

        // 计算相机中心位置 C = -Rᵀ * t
        cv::Mat C = -R.t() * t;
        // 旋转矩阵转四元数
        cv::Vec4d q = rotationMatrixToQuaternion(R);

        // 输出格式：
        // 文件名 焦距 qw qx qy qz Cx Cy Cz 0
        file << imageNames[i] << " " << f << " "
            << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << " "
            << C.at<double>(0) << " " << C.at<double>(1) << " " << C.at<double>(2)
            << " 0\n";
    }

    file << "\n";

    // 写入三维点信息
    file << points3D.size() << "\n";

    for (size_t i = 0; i < points3D.size(); ++i) {
        const cv::Point3d& P = points3D[i];
        const auto& visList = viewIndices[i];

        // ---- 提取颜色（从第一个可见图像取或取平均）----
        cv::Vec3d color(0, 0, 0);
        int count = 0;
        for (size_t j = 0; j < visList.size(); ++j) {
            int imgIdx = visList[j];
            const cv::Mat& img = images[imgIdx];
            // 从 projections2D_all 中获取第 imgIdx 相机上第 i 个三维点的投影坐标
            cv::Point2d pt = projections2D_all[imgIdx][i];
            if (pt.x >= 0 && pt.x < img.cols && pt.y >= 0 && pt.y < img.rows) {
                cv::Vec3b pix = img.at<cv::Vec3b>(cv::Point(pt.x, pt.y));
                color += cv::Vec3d(pix[2], pix[1], pix[0]); // RGB 顺序
                count++;
            }
        }
        if (count > 0) color /= count;
        else color = cv::Vec3d(255, 255, 255); // 没有颜色信息则设白

        // ---- 写入三维点 ----
        file << P.x << " " << P.y << " " << P.z << " "
            << (int)color[0] << " " << (int)color[1] << " " << (int)color[2] << " ";

        // ---- 写入可见性信息 ----
        file << visList.size() << " ";
        for (size_t j = 0; j < visList.size(); ++j) {
            int imgIdx = visList[j];
            const cv::Point2d& pt = projections2D_all[imgIdx][i]; // 从相机 imgIdx 获取第 i 个三维点的投影
            file << imgIdx << " " << j << " " << pt.x << " " << pt.y << " ";
        }
        file << "\n";
    }

    file << "\n0\n";
    file.close();
    std::cout << "NVM 导出完成：scene.nvm" << std::endl;
}