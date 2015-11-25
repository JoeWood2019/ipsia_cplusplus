#ifndef _IPSIA_
#define _IPSIA_

#include <iostream>
#include <string>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

Mat img_gray_RGB2YCbCr(Mat src);
Mat img_grad_mask(Mat grad_mag,Mat grad_x,Mat grad_y,Mat img_sym, double TH_G, double TH_P);
bool isEdge(Mat grad_mag, Mat img_grad_x, Mat img_grad_y, int x, int y);
double linearGM(Mat grad_mag,double xf,double yf,bool isHor);
bool trace_along_gradient(Mat grad_mag, int x, int y, double xf, double yf, double dy, double dx, bool isHor)

Mat img_ipsia(string filename);
#endif