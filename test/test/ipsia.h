#ifndef _IPSIA_
#define _IPSIA_

#include <iostream>
#include <string>
#include <opencv2\opencv.hpp>
#define pi 3.1416926
#define direction_num 3

using namespace cv;
using namespace std;

Mat img_gray_RGB2YCbCr(Mat src,Mat *cb,Mat *cr,int scale);
double min_8_neighbor(Mat image, int i, int j); // mininum in (x:x=2,y:y+2)
double max_8_neighbor(Mat image, int i, int j); // maxnum in (x:x+2,y:y+2)
bool max_value_position_in_array(int *data, int *max_value, int *idx, int array_len);

Mat img_grad_mask(Mat grad_mag,Mat grad_x,Mat grad_y,Mat img_sym, double TH_G, double TH_P);
bool isEdge(Mat grad_mag, Mat img_grad_x, Mat img_grad_y, int x, int y);
double linearGM(Mat grad_mag,double xf,double yf,bool isHor);
bool trace_along_gradient(Mat grad_mag, int x, int y, double dy, double dx, bool isHor);
Mat stickExtract(Mat img_input_sym, Mat img_mask, int scale, bool bp_on);
void edgeProcess(Mat img_sym, Mat *imgH, Mat *edges, Mat mask, Mat gradx, Mat grady, int scale);
void fililRod(Mat *block, Mat *p, int L, int scale, int slope, int TH);

Mat img_ipsia(string filename,int scale);
#endif