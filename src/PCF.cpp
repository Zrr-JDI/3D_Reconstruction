#include"PCF.h"

int DensifyPointCloud()
{
    return system(".\\FUNC\\DensifyPointCloud -i .\\MVS\\scene.mvs -o .\\MVS\\scene.mvs > .\\logs\\DensifyPointCloud.log 2>&1");
}

int ReconstructMesh()
{
    return system(".\\FUNC\\ReconstructMesh -i .\\MVS\\scene.mvs -o .\\MVS\\scene.mvs > .\\logs\\ReconstructMesh.log 2>&1");
}

int RefineMesh()
{
    return system(".\\FUNC\\RefineMesh -i .\\MVS\\scene.mvs -o .\\MVS\\scene.mvs > .\\logs\\RefineMesh.log 2>&1");
}

int TextureMesh()
{
    return system(".\\FUNC\\TextureMesh -i .\\MVS\\scene.mvs -o .\\MVS\\scene.mvs > .\\logs\\TextureMesh.log 2>&1");
}
