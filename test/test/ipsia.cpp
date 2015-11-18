#include "ipsia.h"

Mat img_gray_RGB2YCbCr(Mat src)
{
	Mat img_ycbcr = src.clone();

	cvtColor(src, img_ycbcr, CV_BGR2YCrCb);
	vector<Mat> channels;
	Mat y;
	//Mat cb;
	//Mat cr;
	
	split(img_ycbcr, channels);
	y = channels.at(0);
	//cr = channels.at(1);
	//cb = channels.at(2);

	//imshow("gray_mao", y);
	//waitKey();
	return y;
}

Mat img_ipsia(string filename)
{
	Mat img = imread(filename);
	Mat img_input;
	Mat img_H;

	// just don't input empty image
	if (img.empty())
	{
		cout << "error";
		exit(-1);
	}
	// rgb or gray input image
	if (img.channels() == 1)
	{
		img_input = img.clone();
	}
	else
	{
		img_input = img_gray_RGB2YCbCr(img);
	}

	img_H = img_input.clone();

	// make a first&end-symmetrical input image
	Mat img_input_sym;
	img_input_sym.create(img_input.rows + 2, img_input.cols + 2, CV_8U);
	img_input.copyTo(img_input_sym(Range(1, 1 + img_input.rows), Range(1, 1 + img_input.cols)));// (2:end-1,2:end-1) just the input gray image
	img_input.col(0).copyTo(img_input_sym(Range(1, 1 + img_input.rows), Range(0, 1)));// (2:end-1,1) 
	img_input.col(img_input.cols - 1).copyTo(img_input_sym(Range(1, 1 + img_input.rows), Range(img_input_sym.cols - 1, img_input_sym.cols)));// (2:end,end)
	img_input_sym.row(1).copyTo(img_input_sym.row(0));// (1,1:end)
	img_input_sym.row(img_input_sym.rows - 2).copyTo(img_input_sym.row(img_input_sym.rows - 1));//(end,1:end)

	Mat grad_x, grad_y;
	Mat abs_grad_x;
	Sobel(img_input, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_REPLICATE);
	convertScaleAbs(grad_x, abs_grad_x);

	imshow("gray_mao_sym", abs_grad_x);
	waitKey();
	return img_H;
}