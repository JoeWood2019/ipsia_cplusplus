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
double min_8_neighbor(Mat image, int i, int j)// mininum in (x:x=2,y:y+2)
{
	if (image.depth() != CV_8U)
	{
		cout << "the data type of the image is wrong!" << endl;
		exit(-1);
	}
	double neighbor[8];
	neighbor[0] = image.at<uchar>(i, j);
	neighbor[1] = image.at<uchar>(i, j + 1);
	neighbor[2] = image.at<uchar>(i, j + 2);
	neighbor[3] = image.at<uchar>(i + 1, j);
	neighbor[4] = image.at<uchar>(i + 1, j + 2);
	neighbor[5] = image.at<uchar>(i + 2, j);
	neighbor[6] = image.at<uchar>(i + 2, j + 1);
	neighbor[7] = image.at<uchar>(i + 2, j + 2);

	double min_value;
	min_value = neighbor[0];
	for (int x = 1; x < 8; x++)
	{
		if (neighbor[x] < min_value)
		{
			min_value = neighbor[x];
		}
	}
	return min_value;
}
double max_8_neighbor(Mat image, int i, int j) // maxinum in (x:x+2,y:y+2)
{
	if (image.depth() != CV_8U)
	{
		cout << "the data type of the image is wrong!" << endl;
		exit(-1);
	}
	double neighbor[8];
	neighbor[0] = image.at<uchar>(i, j);
	neighbor[1] = image.at<uchar>(i, j + 1);
	neighbor[2] = image.at<uchar>(i, j + 2);
	neighbor[3] = image.at<uchar>(i + 1, j);
	neighbor[4] = image.at<uchar>(i + 1, j + 2);
	neighbor[5] = image.at<uchar>(i + 2, j);
	neighbor[6] = image.at<uchar>(i + 2, j + 1);
	neighbor[7] = image.at<uchar>(i + 2, j + 2);

	double max_value;
	max_value = neighbor[0];
	for (int x = 1; x < 8; x++)
	{
		if (neighbor[x] > max_value)
		{
			max_value = neighbor[x];
		}
	}
	return max_value;
}
bool max_value_position_in_array(int *data, int *max_value, int *idx) // return max_multi_flag
{
	int array_len = sizeof(data) / sizeof(data[0]);
	// find the first max value and its position
	*max_value = data[0];
	*idx = 0;
	for (int i = 0; i < array_len; i++)
	{
		if (data[i] > *max_value)
		{
			*max_value = data[i];
			*idx = i;
		}
	}
	// max value count
	int max_count = 0;
	for (int i = 0; i < array_len; i++)
	{
		if (data[i] == *max_value)
		{
			max_count++;
		}
	}

	if (max_count == 0)
	{
		return false;
	}
	else
	{
		return true;
	}
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

bool trace_along_gradient(Mat grad_mag, int x,int y, double dy, double dx, bool isHor) // attention!! y= number of the current row, x= current col
{
	int cnt = 0;
	double pre = grad_mag.at<double>(y, x);
	double G = 0,diff=0,xf_next=0,yf_next=0;

	while (true)
	{
		xf_next = x + dx;
		yf_next = y + dy;
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
	double G = 0,diff =0;
	double grad_x, grad_y;
	grad_x = img_grad_x.at<double>(y, x);
	grad_y = img_grad_y.at<double>(y, x);

	// bi-trace from (y,x)
	if (abs(grad_x) > abs(grad_y))
	{
		// trace along x positive
		double dy = grad_y / grad_x;
		double dx = 1;
		flag = trace_along_gradient(grad_mag, x, y, dy, dx, true);
		if (!flag)
		{
			return false;
		}
		 
		// trace along x negative
		dx = -dx;
		dy = -dy;
		flag = trace_along_gradient(grad_mag, x, y, dy, dx, true);
		if (!flag)
		{
			return false;
		}
	}
	else
	{
		// trace along y position
		double dy = 1;
		double dx = grad_x / grad_y;
		flag = trace_along_gradient(grad_mag, x, y, dy, dx, true);
		if (!flag)
		{
			return false;
		}

		// trace along x negative
		dx = -dx;
		dy = -dy;
		flag = trace_along_gradient(grad_mag, x, y, dy, dx, true);
		if (!flag)
		{
			return false;
		}

	}
	return flag;
}

Mat img_grad_mask(Mat grad_mag,Mat grad_x,Mat grad_y, Mat img_sym, double TH_G, double TH_P) // edge extraction
{
	if (grad_mag.depth() != CV_64F)
	{
		cout << "the data type of the grad_mag is wrong!" << endl;
		exit(-1);
	}
	Mat mask;
	mask.create(grad_mag.rows, grad_mag.cols, CV_8U); // 
	int i = 0, j = 0;
	bool flag_edge;

	for (i = 0; i < grad_mag.rows; i++)
	{
		for (j = 0; j < grad_mag.cols; j++)
		{
			mask.at<unsigned char>(i, j) = 0;
			if (grad_mag.at<double>(i, j) < TH_G)
			{
				continue;
			}
			flag_edge = isEdge(grad_mag, grad_x, grad_y, i, j);
			if (flag_edge)
			{
				double min_neighbor = min_8_neighbor(img_sym, i, j);
				double max_neighbor = max_8_neighbor(img_sym, i, j);
				double current_pixel = img_sym.at<uchar>(i, j);
				if ( ((current_pixel - max_neighbor) > TH_P) || ((current_pixel - min_neighbor) < -TH_P) )
				{
					continue;
				}
				mask.at<unsigned char>(i, j) = 255;
			}	
		}
	}
	return mask;
}

Mat stickExtract(Mat img_input_sym, Mat img_mask, int scale, bool bp_on)
{
	int maxIter, minDiff;
	if (bp_on)
	{
		maxIter = 20;
		minDiff = 1;
	}

	double PI90 = pi / 2;
	double PI45 = pi / 4;
	Mat imgH = Mat::zeros(scale*img_mask.rows,scale*img_mask.cols,CV_8U);
	return imgH;
}

void fililRod(Mat *block, Mat *p,int L, int scale, int slope, int TH)
{
	int PL, PR;
	if (abs(slope) > TH) // vertical situation 
	{
		PL = int(ceil(double(scale) / 2));
		if (scale % 2 == 0)
		{
			PR = PL + 1;
		}
		else
		{
			PR = PL;
		}

		for (int i = 0; i < L; i++)
		{
			for (int j = 0; j < scale; j++)
			{
				if (j < PL)
				{
					(*p)(Rect(j, i*scale, 1, scale)) = (*block).at<uchar>(i, 1);
					continue;
				}
				if (j > PR)
				{
					(*p)(Rect(j, i*scale, 1, scale)) = (*block).at<uchar>(i, 3);
					continue;
				}
				(*p)(Rect(j, i*scale, 1, scale)) = (*block).at<uchar>(i, 2);
			}
		}
	}
	else
	{
		Mat pLR = Mat::zeros(L*scale, 1, CV_32S);
		if (slope > 0)
		{
			for (int i = 0; i < scale; i++)
			{
				(*block)(Rect(1, 0, 1, L)).copyTo((*p)(Rect(i, i*L, 1, L)));
				pLR(Rect(0, i*L, 1, L)) = i;
			}
		}
		else
		{
			for (int i = 1; i < scale; i++)
			{
				(*block)(Rect(1, 0, 1, L)).copyTo((*p)(Rect(scale-i, i*L, 1, L)));
				pLR(Rect(0, i*L, 1, L)) = scale - i;
			}
		}
		for (int i = 0; i < scale*L; i++)
		{
			for (int j = 0; j < scale; j++)
			{
				if (j < pLR.at<int>(i, 1))
				{
					(*p).at<uchar>(i, j) = (*block).at<uchar>(int(ceil(double(i) / double(scale) )), 0);
					continue;
				}
				if (j > pLR.at<int>(i, 1))
				{
					(*p).at<uchar>(i, j) = (*block).at<uchar>(int(ceil(double(i) / double(scale))), 2);
					continue;
				}
			}
		}
	}
}

Mat edgeProcess(Mat img_sym, Mat *imgH, Mat *edges, Mat mask, Mat gradx, Mat grady, int scale)
{
	int H = mask.rows, W = mask.cols;
	int i = 0, j = 0, k = 0;

	int TH_v = 6;
	int head_y = -1, head_x = -1, tail_y = -1, tail_x = -1;
	int sig[3];
	int len = 0, slope = 0;

	double gx, gy;

	bool flag_mask=false, flag_mask_pre=false;

	for (j = 0; j < W; j++)
	{
		for (i = 0; i < H; i++)
		{
			if ((mask.at<uchar>(i, j) == 0 && len == 0)) // the value in mask(i,j) is negative
			{
				flag_mask_pre = false;
				continue;
			}
			if ((j == 0) || (j == W)) // boundary situation
			{
				flag_mask = (mask.at<uchar>(i, j) > 0);
			}
			else // mask(i,j) - ( mask(i,j) & mask(i,j-1) & mask(i,j+1) )
			{
				flag_mask = (mask.at<uchar>(i, j) > 0) ^ ((mask.at<uchar>(i, j) > 0) && (mask.at<uchar>(i, j - 1) > 0) && (mask.at<uchar>(i, j + 1) > 0));
			}

			// ----------------------------------------------------------------------------------------------
			if (!flag_mask && len == 0) // rod of 3 pixels long is not horizonal	
			{
				flag_mask_pre = flag_mask;
				continue;
			}

			if (len == 0) // the start of rod
			{
				flag_mask_pre = flag_mask;
				len = 1;
				head_y = i;
				head_x = j;
				tail_y = i;
				tail_x = j;
				for (k = 0; k < 3; k++) // initialize the sig which marks three different directions(situations) 
				{
					sig[k] = 0;
				}
				// obtain the graduation information
				gy = grady.at<double>(i, j);
				gx = gradx.at<double>(i, j);
				// three directions
				if (abs(gy) > 0)
				{
					if (abs(gy) > 1)
					{
						sig[1] = sig[1] + 1;
					}
					else
					{
						if (gy*gx > 0)
						{
							sig[0] = sig[0] + 1;
						}
						else
						{
							sig[2] = sig[2] + 1;
						}
					}
				}
			}
			else
			{
				// the flag_mask positive pixels is continue in y coordinate and the same in x coordinate
				////bool flag = ((mask.at<uchar>(i, j) == mask.at<uchar>(i - 1, j)) && (mask.at<uchar>(i, j) == mask.at<uchar>(i, j - 1)));
				bool flag = flag_mask && flag_mask_pre;
				if (flag && i != 0) // i!=0 prevent j:j -> j+1, i:H-1 -> 0
				{
					len = len + 1;
					tail_y++;
					// obtain the graduation information
					gy = grady.at<double>(i, j);
					gx = gradx.at<double>(i, j);
					// three directions
					if (abs(gy) > 0)
					{
						if (abs(gy) > 1)
						{
							sig[1] = sig[1] + 1;
						}
						else
						{
							if (gy*gx > 0)
							{
								sig[0] = sig[0] + 1;
							}
							else
							{
								sig[2] = sig[2] + 1;
							}
						}
					}
					continue;
				}

				// deal with the whole rod
				if (len > TH_v)
				{
					slope = 10;
				}
				else
				{
					bool corner[4]; // 4 corner around the rod
					if (head_y == 0 || head_x == 0) // left-top 
					{
						corner[0] = false;
					}
					else
					{
						corner[0] = (mask.at<uchar>(head_y-1, head_x-1) > 0);
					}
					if (head_y == 0 || head_x == W) // right-top 
					{
						corner[1] = false;
					}
					else
					{
						corner[1] = (mask.at<uchar>(head_y - 1, head_x + 1) > 0);
					}
					if (tail_y == H || tail_x == 0) // left-bottom
					{
						corner[2] = false;
					}
					else
					{
						corner[2] = (mask.at<uchar>(tail_y+ 1, tail_x - 1) > 0);
					}
					if (tail_y == H || tail_x == W) // right-bottom
					{
						corner[3] = false;
					}
					else
					{
						corner[3] = (mask.at<uchar>(tail_y + 1, tail_x + 1) > 0);
					}
					if (corner[0] || corner[1] || corner[2] || corner[3])
					{
						int max_value_sig = 0, idx = 0;
						bool max_value_multi_flag = max_value_position_in_array(sig, &max_value_sig, &idx);
						if (max_value_multi_flag)
						{
							slope = 0;
						}
						else
						{
							switch (idx)
							{
							case 0:
								slope = -len;
								break;
							case 1:
								slope = 10;
								break;
							case 2:
								slope = len;
								break;
							default:
								slope = 0;
							}
						}
					}
					else
					{
						slope = 0;
					}
				}

				// handle rod
				if (slope != 0)
				{
					(*edges)(Rect(head_x, head_y, tail_x - head_x+1, tail_y - head_y+1)) = 255;
					Mat block;
					block.create(tail_y - head_y + 1, tail_x - head_x + 3, CV_8U);
					img_sym(Rect(head_x - 2, head_y - 1, tail_x - head_x + 3, tail_y - head_y + 1)).copyTo(block);
					int L = block.rows;
					Mat Hpatch = Mat::zeros(scale*L, scale, CV_8U);
					fililRod(&block, &Hpatch, L, scale, slope, TH_v);
					Hpatch.copyTo((*imgH)(Rect(head_x*scale, head_y*scale, (tail_x - head_x+1)*scale, (tail_y - head_y+1)*scale)));
					namedWindow("block", WINDOW_NORMAL);
					imshow("block", Hpatch);
					waitKey();
				}

				// prepare for next rod
				len = 0;
				flag_mask_pre = flag_mask;
			}
		}
	}
}

Mat img_ipsia(string filename,int scale)
{
	Mat img = imread(filename);
	Mat img_input;

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

	Mat img_H,edges;
	img_H.create(img_input.rows * scale, img_input.cols * scale, CV_8U);
	edges.create(img_input.rows * scale, img_input.cols * scale, CV_8U);

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

	//Mat _img_dest;
	//double ostu_threh = threshold(img_input, _img_dest, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);// get the ostu threshold for canny
	//Mat img_canny;
	//Canny(img_input, img_canny, ostu_threh*0.5, ostu_threh, 3);
	//// use a mask to delete the isolated pixels
	//Mat kern_clean = (Mat_<char>(3, 3) << 1, 1, 1, 1, 0, 1, 1, 1, 1);
	//Mat img_canny_clean_mask;
	//filter2D(img_canny, img_canny_clean_mask, CV_16U, kern_clean); // after filter, the isolated pixel will be 0
	//img_canny_clean_mask = Mat_<uchar>(abs(grad_y));
	//Mat img_canny_clean; // delete isolated pixel and get a clean canny image
	//img_canny.copyTo(img_canny_clean, img_canny_clean_mask);

	// get the mask that indicate the edge area------------------------------------------------------------------
	double TH_G = 100;
	double TH_P = 50;
	Mat img_mask = img_grad_mask(grad_magnitude, grad_x, grad_y, img_input_sym, TH_G, TH_P);
	  // delete isolated pixel and get a clean mask image
	Mat kern_clean = (Mat_<char>(3, 3) << 1, 1, 1, 1, 0, 1, 1, 1, 1);
	Mat img_mask_clean_temp;
	filter2D(img_mask, img_mask_clean_temp, CV_8U, kern_clean); // after filter, the isolated pixel will be 0
	Mat img_mask_clean; 
	img_mask.copyTo(img_mask_clean, img_mask_clean_temp);

	// 
	edgeProcess(img_input_sym, &img_H, &edges, img_mask_clean, grad_x, grad_y, scale);

	//Mat img_show;
	//normalize(img_mask_clean, img_show, 0, 255, NORM_MINMAX);
	namedWindow("mao", WINDOW_NORMAL);
	imshow("mao", img_mask_clean);
	waitKey();

	return img_H;
} 