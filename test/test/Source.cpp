#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include "ipsia.h"
using namespace cv;
using namespace std;
int main()
{
	Mat img_result = img_ipsia("C04_4_LR.png");
	
	imshow("RGB_mao", img_result);
	waitKey();
	return 0;
}