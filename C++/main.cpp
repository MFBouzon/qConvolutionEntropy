#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>

#include "Firefly_Optimization.h"
#include "multilimiarizacao.h"

using namespace cv;
using namespace std;

void getEntropy(Mat src_gray_small_2, Mat segment)
{
	int intensity;


	int Lin = src_gray_small_2.rows;
	int Col = src_gray_small_2.cols;

	double EntropyRegion = 0;
	double Total = 0;
	double T = 0;

	int d = 3;
	int r, c;
	for (int y = d; y < Lin - d; y++)
		for (int x = d; x < Col - d; x++)
		{
			Mat Histgray = Mat::zeros(1, 256, CV_32F);
			T = 0;
			for (r = y - d; r < y + d; r++)
				for (c = x - d; c < x + d; c++)
				{ 
					intensity = (int)src_gray_small_2.at<uchar>(r, c);
					Histgray.at<float>(intensity) = Histgray.at<float>(intensity) + 1.0;
					T = T + 1;
				}
			// calcula a entropia
			double S = 0;
			for (int L = 0; L < 256; L++)
			{
				Histgray.at<float>(L) = Histgray.at<float>(L) / T;
				double Pintesity = Histgray.at<float>(L);
				if (Pintesity != 0)
					S = S + Pintesity * log2(Pintesity);
			}

			segment.at<double>(y, x) = -S;
			
		}

}

void getqEntropy(Mat src, Mat &dst0)
{

	//imshow("entropy", src);
	//waitKey(0);

	int intensity;

	int Lin = src.rows;
	int Col = src.cols;

	double Total = 0;
	double T = 0;
	double q = 0.5;

	int d = 5;
	int r, c;

	Mat entropy_temp = Mat::zeros(Lin, Col, CV_32F);
	double max = 0, min = 99999999;
	for (int y = d; y < Lin - d; y++)
		for (int x = d; x < Col - d; x++)
		{
			Mat Histgray = Mat::zeros(1, 256, CV_32F);
			T = 0;
			for (r = y - d; r < y + d; r++)
				for (c = x - d; c < x + d; c++)
				{
					intensity = src.at<uchar>(r, c);
					Histgray.at<float>(intensity) = Histgray.at<float>(intensity) + 1.0;
					T = T + 1;
				}

			// calcula a q-entropia
			double S = 0;
			for (int L = 0; L < 256; L++)
			{
				Histgray.at<float>(L) = Histgray.at<float>(L) / T;
				double Pintesity = Histgray.at<float>(L);
				if (Pintesity != 0)
					S = S + pow(Pintesity, q);
			}
			
			if (q != 1)
				S = (1 - S) / (q - 1);
			else
				S = 1 - S;
			//S = S / log2(255);
			entropy_temp.at<float>(y, x) = S;
			Total = Total + S;
			//printf("S = %.3f\n",S);
			if (S > max)
				max = S;
			if (S < min)
				min = S;
		}
	
	for(int y = d+1; y < Lin;y++)	
		for (int x = d+1; x < Col; x++) {
			entropy_temp.at<float>(y, x) /= max;
			entropy_temp.at<float>(y, x) *= 255;
			
		}

	printf("%.3f %.3f\n", max, min);
	dst0 = entropy_temp.clone();

}

int main() {
	Mat img = imread("praia.jpg", 0);
	Mat result(img.rows, img.cols, CV_64F);
	Mat result2(img.rows, img.cols, CV_8U);
	Mat result3;
	
	vector<unsigned int> bests = beststhresholds(img, 2, 100, 100);
	result3 = MultiLim2(img, bests, 0);

	getEntropy(img, result);
	result.convertTo(result2, CV_8U);
	equalizeHist(result2, result2);
	imshow("Original", img);
	imshow("Teste", result);
	imshow("Teste2", result2);
	imshow("Firefly", result3);

	waitKey();

}

