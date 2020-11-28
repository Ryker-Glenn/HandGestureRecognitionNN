#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>

constexpr auto MAX_KERNEL_LENGTH	= 11;
constexpr auto MAX_COLOR			= 255;
constexpr auto THRESH1				= 0.3;
constexpr auto C1					= 6.5025;
constexpr auto C2					= 58.5225;

using namespace std;
using namespace cv;

class Gesture {
public:
	int prog = 0;
	double prev_sim = 0;
	Gesture();
	void capture();
private:
	void update_mhi(Mat&, Mat&);
	void update_progress(Mat&);	
	Mat add(Mat&, Mat&);

	/* utility functions used to determine structural 
	similarity between initial frame and merged frame */
	double avg(int, int);
	double sq(double);

	// Structural similarity calculations between an initial image and the merged image
	double ssim(const Mat&, const Mat&);

	// Calculates variances of img1 and img2, as well as covariance to save some time
	double calc_variances_similarity(const Mat&, const Mat&, double, double);
};