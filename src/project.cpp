#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "../include/Model.h"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    std::string datasetPath = "../../../pose_estimation_dataset";

    // you can also set the path of the dataset by command line
    if (argc == 2)
        datasetPath = argv[1];

    Model *can = new Model(datasetPath, "can");
    Model *driller = new Model(datasetPath, "driller");
    Model *duck = new Model(datasetPath, "duck");

    can->start();
    driller->start();
    duck->start();

    return 0;
}