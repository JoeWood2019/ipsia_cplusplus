#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include "ipsia.h"
using namespace cv;
using namespace std;
int main()
{
	Mat img_result = img_ipsia("D03.bmp",3);
	
	imshow("RGB_mao", img_result);
	waitKey();
	return 0;
}