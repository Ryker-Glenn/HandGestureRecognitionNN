#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/optflow.hpp>

#define DELAY_BLUR			100
#define MAX_KERNEL_LENGTH	11
#define MAX_BV				255
#define BI					1
#define MOD					255
#define THRESH1				0.5
#define THRESH2				0
#define C1					6.5025		// Calculated as (k_1 * L)^2 where k_1 = 0.01 and L is 2^8 - 1
#define C2					58.5225		// Calculated as (k_2 * L)^2 where k_2 = 0.03 and L is 2^8 - 1

using namespace std;
using namespace cv;
using namespace cv::motempl;

class Gesture {
public:
	int prog = 0;
	double prev_sim = 0;
	const double MHI_DURATION = 5;
	const double MAX_TIME_DELTA = 0.5;
	const double MIN_TIME_DELTA = 0.05;
	VideoCapture cap;
	vector<Mat> buf; // img ring buffer
	Gesture();
	void capture();
private:
	void update_mhi(const Mat&, Mat&);
	void update_progress(const Mat&, Mat&);
	void mod(const Mat&, Mat&);
	Mat add(Mat&, Mat&, Mat&);
	double structural_similarity(Mat&, Mat&);

	double average(int, int);
	long variance(const Mat&, double, int);
	long covariance(const Mat&, const Mat&, double, double, int);
	long square(long);
	long ssim(long, long, long, long, long);
};