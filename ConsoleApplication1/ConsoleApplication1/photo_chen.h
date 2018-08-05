#include <iostream>
#include "opencv/cv.h"
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
using namespace std;
using namespace cv;
#pragma once
class photo_chen
{

public:
	photo_chen();
	~photo_chen();
	cv::Mat R_chen(double x,double y,double z);
};

