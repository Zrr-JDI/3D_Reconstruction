#include"PCF.h"
#include <cstdlib>
#include <filesystem>
#include <string>
#include <iostream>

namespace fs = std::filesystem;

bool EnsureLogDirectory() {
    try {
        if (!fs::exists("logs")) {
            fs::create_directories("logs");
            std::cout << "创建日志目录: logs" << std::endl;
        }

        std::vector<std::string> subDirs = {
            "logs\\InterfaceVisualSFM_logs",
            "logs\\DensifyPointCloud_logs",
            "logs\\ReconstructMesh_logs",
            "logs\\RefineMesh_logs",
            "logs\\TextureMesh_logs"
        };

        for (const auto& dir : subDirs) {
            if (!fs::exists(dir)) {
                fs::create_directories(dir);
                std::cout << "创建日志子目录: " << dir << std::endl;
            }
        }

        std::cout << "确保所有日志目录存在完成" << std::endl;
        return true;
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "创建日志目录失败: " << e.what() << std::endl;
        return false;
    }
}

void CleanCurrentDirLogs() {
    try {
        for (const auto& entry : fs::directory_iterator(".")) {
            if (entry.path().extension() == ".log") {
                fs::remove(entry.path());
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "清理当前目录日志文件失败: " << e.what() << std::endl;
    }
}

int InterfaceVisualSFM() {
    int result = system(".\\FUNC\\InterfaceVisualSFM.exe -i .\\MVS\\scene.nvm -o .\\MVS\\scene.mvs > .\\logs\\InterfaceVisualSFM_logs\\InterfaceVisualSFM.log 2>&1");
    if (result != 0) {
        std::cerr << "InterfaceVisualSFM 执行失败，检查 logs/InterfaceVisualSFM_logs/InterfaceVisualSFM.log" << std::endl;
    }
    CleanCurrentDirLogs();
    return result;
}

int DensifyPointCloud() {
    int result = system(".\\FUNC\\DensifyPointCloud.exe -i .\\MVS\\scene.mvs -o .\\MVS\\scene.mvs > .\\logs\\DensifyPointCloud_logs\\DensifyPointCloud.log 2>&1");
    if (result != 0) {
        std::cerr << "DensifyPointCloud 执行失败，检查 logs/DensifyPointCloud_logs/DensifyPointCloud.log" << std::endl;
    }
    CleanCurrentDirLogs();
    return result;
}

int ReconstructMesh() {
    int result = system(".\\FUNC\\ReconstructMesh.exe -i .\\MVS\\scene.mvs -o .\\MVS\\scene.mvs > .\\logs\\ReconstructMesh_logs\\ReconstructMesh.log 2>&1");
    if (result != 0) {
        std::cerr << "ReconstructMesh 执行失败，检查 logs/ReconstructMesh_logs/ReconstructMesh.log" << std::endl;
    }
    CleanCurrentDirLogs();
    return result;
}

int RefineMesh() {
    int result = system(".\\FUNC\\RefineMesh.exe -i .\\MVS\\scene.mvs -o .\\MVS\\scene.mvs > .\\logs\\RefineMesh_logs\\RefineMesh.log 2>&1");
    if (result != 0) {
        std::cerr << "RefineMesh 执行失败，检查 logs/RefineMesh_logs/RefineMesh.log" << std::endl;
    }
    CleanCurrentDirLogs();
    return result;
}

int TextureMesh() {
    int result = system(".\\FUNC\\TextureMesh.exe -i .\\MVS\\scene.mvs -o .\\MVS\\scene.mvs > .\\logs\\TextureMesh_logs\\TextureMesh.log 2>&1");
    if (result != 0) {
        std::cerr << "TextureMesh 执行失败，检查 logs/TextureMesh_logs/TextureMesh.log" << std::endl;
    }
    CleanCurrentDirLogs();
    return result;
}
