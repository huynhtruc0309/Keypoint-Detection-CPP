#pragma once
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <tuple>

using namespace cv;
using namespace std;

Mat detectHarrist(Mat img, int threshold, int k);

Mat calLoG(float sigma);
vector<Mat> LoGConvolve(Mat img);
Mat detectBlob(Mat img, int threshold);

Mat detectDOG(Mat img);
double matchBySIFT(Mat img1, Mat img2, int detector);