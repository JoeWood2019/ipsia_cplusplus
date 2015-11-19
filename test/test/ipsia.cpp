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
	Mat abs_grad_x,abs_grad_y;
	Sobel(img_input, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_REPLICATE);
	Sobel(img_input, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_REPLICATE);
	convertScaleAbs(grad_x, abs_grad_x);
	abs_grad_y = Mat_<uchar>(abs(grad_y));
	
	Mat _img_dest;
	double ostu_threh = threshold(img_input, _img_dest, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);// get the ostu threshold for canny
	Mat img_canny;
	Canny(img_input, img_canny, ostu_threh*0.5, ostu_threh, 3);
	Mat kern_clean = (Mat_<char>(3, 3) << 1, 1, 1, 1, 0, 1, 1, 1, 1);
	Mat img_canny_clean_mask;
	filter2D(img_canny, img_canny_clean_mask, CV_16U, kern_clean);
	img_canny_clean_mask = Mat_<uchar>(abs(grad_y));
	Mat img_canny_clean;// delete isolated pixel and get a clean canny image
	img_canny.copyTo(img_canny_clean, img_canny_clean_mask);

	imshow("gray_mao_sym_canny",img_canny);
	waitKey();

	return img_H;
}