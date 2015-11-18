#ifndef _IPSIA_
#define _IPSIA_

#include <iostream>
#include <string>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

Mat img_gray_RGB2YCbCr(Mat src);
Mat img_ipsia(string filename);

#endif