#include "ipsia.h"

Mat img_gray_RGB2YCbCr(Mat src)
{
	Mat img_ycbcr = src.clone();

	cvtColor(src, img_ycbcr, CV_BGR2YCrCb);
	vector<Mat> channels;
	Mat y;

	split(img_ycbcr, channels);
	y = channels.at(0);

	//Mat cb;
	//Mat cr;
	//cr = channels.at(1);
	//cb = channels.at(2);

	//imshow("gray_mao", y);
	//waitKey();

	return y;
}

// distance along gradient (linear)
double linearGM(Mat grad_mag, double xf, double yf, bool isHor)
{
	double G = 0; // result
	int H = grad_mag.rows;
	int W = grad_mag.cols;
	int xi = int(floor(xf)); // the next pixel position
	int yi = int(floor(yf));

	if (isHor) // if deal with the horizen rod
	{
		if (xi<0 || xi>W-1) // the next pixel's x coordinate beyond the bound
		{
			return G = -1;
		}
		double dy = yf - yi; // double - int
		if (dy < 1e-10) // occasionally yf is a int 
		{
			if (yi<0 || yi>H-1) // the next pixel's y coordinate beyond the bound
			{
				return G = -1;
			}
			else // return the next pixel's gradient magnitude
			{
				return G = grad_mag.at<double>(yi, xi);
			}
		}
		else // usually yf is a double 
		{
			if (yi<-1 || yi>H - 1) // the next pixel's y coordinate far beyond the bound
			{
				return G = -1;
			}
			else
			{
				if (yi == -1) // the next pixel's y coordinate just beyond the bound
				{
					return G = grad_mag.at<double>(yi+1,xi);
				}
				if (yi == H - 1) // the next pixel's y coordinate just the bound
				{
					return G = grad_mag.at<double>(yi, xi);
				}
				// linear interpolation the gradient of (yf,xf)
				return G = grad_mag.at<double>(yi, xi) + ( grad_mag.at<double>(yi + 1, xi) - grad_mag.at<double>(yi, xi) )*dy;
			}
		}
	}
	// vertical situation just the same
	else
	{
		if (yi<0 || yi>H - 1)
		{
			return G = -1;
		}
		double dx = xf - xi;
		if (dx < 1e-10)
		{
			if (xi<0 || xi>W - 1)
			{
				return G = -1;
			}
			else
			{
				return G = grad_mag.at<double>(yi, xi);
			}
		}
		else
		{
			if (xi<-1 || xi>W - 1)
			{
				return G = -1;
			}
			else
			{
				if (yi == -1)
				{
					return G = grad_mag.at<double>(yi, xi + 1);
				}
				if (xi == W - 1)
				{
					return G = grad_mag.at<double>(yi, xi);
				}
				return G = grad_mag.at<double>(yi, xi) + ( grad_mag.at<double>(yi, xi + 1) - grad_mag.at<double>(yi, xi) )*dx;
			}
		}
	}
}

bool trace_along_gradient(Mat grad_mag, int x,int y,double xf, double yf, double dy, double dx, bool isHor)
{
	int cnt = 0;
	double pre = grad_mag.at<double>(y, x);
	double G = 0,diff=0,xf_next=0,yf_next=0;

	while (true)
	{
		xf_next = xf + dx;
		yf_next = yf + dy;
		G = linearGM(grad_mag, xf_next, yf_next, true); // the gradient magnitude in (xf,yf) (a linear interpolation approximate) 
		if (G == -1)
		{
			break;
		}
		diff = pre - G;
		if (diff > 0)
		{
			cnt++;
			pre = G;
		}
		else
		{
			break;
		}
	}
	if (cnt == 0)
	{
		return false;
	}
	else
	{
		return true;
	}
}

bool isEdge(Mat grad_mag, Mat img_grad_x, Mat img_grad_y, int y, int x) // attention!! y= number of the current row, x= current col
{
	bool flag = false;
	if (grad_mag.at<double>(y, x) < 1e-10)
	{
		return flag;
	}
	// some preparation
	double xf = x;
	double yf = y;
	double G = 0,diff =0;

	double grad_x, grad_y;
	grad_x = img_grad_x.at<double>(y, x);
	grad_y = img_grad_y.at<double>(y, x);
	if (abs(grad_x) > abs(grad_y))
	{
		// trace along x positive
		double dy = grad_y / grad_x;
		double dx = 1;
		flag = trace_along_gradient(grad_mag, x, y, xf, yf, dy, dx, true);
		if (!flag)
		{
			return false;
		}
		 
		// trace along x negative
		dx = -dx;
		dy = -dy;
		xf = x;
		yf = y;
		flag = trace_along_gradient(grad_mag, x, y, xf, yf, dy, dx, true);
		if (!flag)
		{
			return false;
		}
	}
	else
	{

	}
	return flag;
}

Mat img_grad_mask(Mat grad_mag,Mat grad_x,Mat grad_y, Mat img_sym, double TH_G, double TH_P)
{
	if (grad_mag.depth() != 6)
	{
		cout << "the data type of the grad_mag is wrong!" << endl;
		exit(-1);
	}
	Mat mask;
	mask.create(grad_mag.rows, grad_mag.cols, CV_8U); // 
	int i = 0, j = 0;
	for (i = 0; i < grad_mag.rows; i++)
	{
		for (j = 0; j < grad_mag.cols; j++)
		{
			mask.at<unsigned char>(i, j) = 0;
			if (grad_mag.at<double>(i, j) < TH_G)
			{
				continue;
			}
			mask.at<unsigned char>(i, j) = 255;
		}
	}
	return mask;
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
	img_input_sym.row(1).copyTo(img_input_sym.row(0)); // (1,1:end)
	img_input_sym.row(img_input_sym.rows - 2).copyTo(img_input_sym.row(img_input_sym.rows - 1));//(end,1:end)

	// use sobel to get gx gy gm ag
	Mat grad_x, grad_y;
	Sobel(img_input, grad_x, CV_64F, 1, 0, 3, 1, 1, BORDER_REPLICATE);
	Sobel(img_input, grad_y, CV_64F, 0, 1, 3, 1, 1, BORDER_REPLICATE);
	Mat grad_x_square, grad_y_square;	
	pow(grad_x, 2.0, grad_x_square);
	pow(grad_y, 2.0, grad_y_square);

	Mat grad_magnitude;
	grad_magnitude = grad_x_square + grad_y_square;
	sqrt(grad_magnitude, grad_magnitude);
	//Mat grad_angle;
	//atan2(grad_y, grad_x, grad_angle);

	Mat _img_dest;
	double ostu_threh = threshold(img_input, _img_dest, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);// get the ostu threshold for canny
	Mat img_canny;
	Canny(img_input, img_canny, ostu_threh*0.5, ostu_threh, 3);
	// use a mask to delete the isolated pixels
	Mat kern_clean = (Mat_<char>(3, 3) << 1, 1, 1, 1, 0, 1, 1, 1, 1);
	Mat img_canny_clean_mask;
	filter2D(img_canny, img_canny_clean_mask, CV_16U, kern_clean); // after filter, the isolated pixel will be 0
	img_canny_clean_mask = Mat_<uchar>(abs(grad_y));
	Mat img_canny_clean; // delete isolated pixel and get a clean canny image
	img_canny.copyTo(img_canny_clean, img_canny_clean_mask);

	double TH_G = 100;
	double TH_P = 50;
	Mat img_mask = img_grad_mask(grad_magnitude, grad_x, grad_y, img_input_sym, TH_G, TH_P);
	//Mat img_show;
	//normalize(grad_magnitude, img_show, 0, 1, NORM_MINMAX);
	imshow("gray_sobel_mag", img_mask);
	waitKey();

	return img_H;
}