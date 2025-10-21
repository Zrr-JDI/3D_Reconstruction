#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include"PCF.h"
using namespace std;
using namespace cv;

int main()
{
    //表面重建部分
    /*int ret = DensifyPointCloud();
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
    }*/
	return 0;
}