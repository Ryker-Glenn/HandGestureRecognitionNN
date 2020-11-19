#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>

#define DELAY_BLUR 100
#define MAX_KERNEL_LENGTH 11
#define MAX_BV 255
#define BI 1

using namespace std;
using namespace cv;

class Gesture {
public:
	int progress = 0;
	Mat merged_frame, init_frame;
	VideoCapture cap;
	Gesture();
	void capture();
private:
	Mat modulo(Mat, int);
	void update_mhi(Mat);
	Mat add(Mat, Mat);
};