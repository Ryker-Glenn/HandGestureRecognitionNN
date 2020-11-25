#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include "Similarity.h"

#define DELAY_BLUR			100
#define MAX_KERNEL_LENGTH	11
#define MAX_BV				255
#define BI					1
#define MOD					255
#define THRESH1				0.5
#define THRESH2				0

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
	void update_mhi(const Mat&, Mat&);
	void update_progress(const Mat&, Mat&);
	void mod(const Mat&, Mat&);
	Mat add(Mat&, Mat&, Mat&);
};