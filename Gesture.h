#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>

#define MAX_KERNEL_LENGTH	11
#define MAX_COLOR			255
#define THRESH1				0.3
#define C1					6.5025
#define C2					58.5225

using namespace std;
using namespace cv;

class Gesture {
public:
	int prog = 0;
	double prev_sim = 0;
	VideoCapture cap;
	Gesture();
	void capture();
private:
	void update_mhi(Mat&, Mat&);
	void update_progress(Mat&);	
	void add(Mat&, Mat&);

	/* utility functions used to determine structural 
	similarity between initial frame and merged frame */
	double avg(int, int);
	double sq(double);

	// calculates expectation of a pixel from its mean
	double variance(const Mat&, double, int);

	// calculates expected value of the product of x/y's deviations from their individual means
	double covariance(const Mat&, const Mat&, double, double, int);

	// Structural similarity calculations between an initial image and the merged image
	double ssim(const Mat&, const Mat&);

	// does the actual important calculations
	double calc_ssim(double, double, double, double, double);
};