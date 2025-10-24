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
            std::cout << "������־Ŀ¼: logs" << std::endl;
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
                std::cout << "������־��Ŀ¼: " << dir << std::endl;
            }
        }

        std::cout << "ȷ��������־Ŀ¼�������" << std::endl;
        return true;
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "������־Ŀ¼ʧ��: " << e.what() << std::endl;
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
        std::cerr << "����ǰĿ¼��־�ļ�ʧ��: " << e.what() << std::endl;
    }
}

int InterfaceVisualSFM() {
    int result = system(".\\FUNC\\InterfaceVisualSFM.exe -i .\\MVS\\scene.nvm -o .\\MVS\\scene.mvs > .\\logs\\InterfaceVisualSFM_logs\\InterfaceVisualSFM.log 2>&1");
    if (result != 0) {
        std::cerr << "InterfaceVisualSFM ִ��ʧ�ܣ���� logs/InterfaceVisualSFM_logs/InterfaceVisualSFM.log" << std::endl;
    }
    CleanCurrentDirLogs();
    return result;
}

int DensifyPointCloud() {
    int result = system(".\\FUNC\\DensifyPointCloud.exe -i .\\MVS\\scene.mvs -o .\\MVS\\scene.mvs > .\\logs\\DensifyPointCloud_logs\\DensifyPointCloud.log 2>&1");
    if (result != 0) {
        std::cerr << "DensifyPointCloud ִ��ʧ�ܣ���� logs/DensifyPointCloud_logs/DensifyPointCloud.log" << std::endl;
    }
    CleanCurrentDirLogs();
    return result;
}

int ReconstructMesh() {
    int result = system(".\\FUNC\\ReconstructMesh.exe -i .\\MVS\\scene.mvs -o .\\MVS\\scene.mvs > .\\logs\\ReconstructMesh_logs\\ReconstructMesh.log 2>&1");
    if (result != 0) {
        std::cerr << "ReconstructMesh ִ��ʧ�ܣ���� logs/ReconstructMesh_logs/ReconstructMesh.log" << std::endl;
    }
    CleanCurrentDirLogs();
    return result;
}

int RefineMesh() {
    int result = system(".\\FUNC\\RefineMesh.exe -i .\\MVS\\scene.mvs -o .\\MVS\\scene.mvs > .\\logs\\RefineMesh_logs\\RefineMesh.log 2>&1");
    if (result != 0) {
        std::cerr << "RefineMesh ִ��ʧ�ܣ���� logs/RefineMesh_logs/RefineMesh.log" << std::endl;
    }
    CleanCurrentDirLogs();
    return result;
}

int TextureMesh() {
    int result = system(".\\FUNC\\TextureMesh.exe -i .\\MVS\\scene.mvs -o .\\MVS\\scene.mvs > .\\logs\\TextureMesh_logs\\TextureMesh.log 2>&1");
    if (result != 0) {
        std::cerr << "TextureMesh ִ��ʧ�ܣ���� logs/TextureMesh_logs/TextureMesh.log" << std::endl;
    }
    CleanCurrentDirLogs();
    return result;
}
