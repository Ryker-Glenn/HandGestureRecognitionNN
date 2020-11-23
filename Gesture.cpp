#include "Gesture.h"

Gesture::Gesture() {}

void Gesture::capture() {
	Mat frame, fgMask, fgBlur, fgThresh, merged_frame;
	Ptr<BackgroundSubtractor> pBackSub;
	pBackSub = createBackgroundSubtractorMOG2(false);			// trained NN from OpenCV used to remove everything not moving in the background
	cap.open("gesture_tests/swipe_right.mp4");
	/*cap.open(0);*/ // Open webcam
	while (true) {
		cap >> frame;
		if (frame.empty()) { break; }
		pBackSub->apply(frame, fgMask);							// apply background remover to current video
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		for (int i = 1; i < MAX_KERNEL_LENGTH; i += 2)
			medianBlur(fgMask, fgBlur, i);						// For the kernel size, apply opencv's median blur function to frame with removed
																// background
		threshold(fgBlur, fgThresh, 0, MAX_BV, BI);				// apply binary inverted threshold to the blurred frame
		/*imshow("BI Blur FG Mask", fgThresh);*/
		update_mhi(fgThresh, merged_frame);

		imshow("mhi", merged_frame);
		// show video frame-by-frame
		//get the input from the keyboard
		if (waitKey(10) >= 0)
            break;
	}
}

void Gesture::update_mhi(const Mat& img, Mat& dst) {
	if (dst.size() != img.size()) {
		dst = Mat::zeros(img.size(), CV_8U);
		dst.setTo(255);
	}
	else {
		Mat after_mod, progress_set, init_frame(img);
		double similarity;
		update_progress(img, progress_set);
		mod(img, after_mod);
		add(after_mod, dst);
		similarity = structural_similarity(init_frame, dst);
		if (abs(similarity - prev_sim) > THRESH1) {
			prog++;
		}
		prev_sim = similarity;
	}
}

void Gesture::mod(const Mat& img, Mat& after) {
	after.create(img.rows, img.cols, img.type());
	for (int y = 0; y < img.cols * img.rows; y++) {
		after.at<uchar>(y) = img.at<uchar>(y) % MOD;
	}
}

void Gesture::add(const Mat& mod, const Mat& merged) {
	Mat src1(mod), src2(merged);
}

double Gesture::structural_similarity(Mat& init, Mat& merged) {
	int px_init = 0, px_merged = 0, total = 0;
	double avg_init, avg_merged;
	long variance_init, variance_merged, covar;
	for (int y = 0; y < init.cols * init.rows; y++) {
		total++;
		px_init += init.at<uchar>(y);
		px_merged += merged.at<uchar>(y);
	}
	avg_init = average(px_init, total);
	avg_merged = average(px_merged, total);
	variance_init = variance(init, avg_init, total);
	variance_merged = variance(merged, avg_merged, total);
	covar = covariance(init, merged, avg_init, avg_merged, total);

	return ssim(avg_init, avg_merged, covar, variance_init, variance_merged);
}

void Gesture::update_progress(const Mat& img, Mat& progress) {
	const int nChannels = img.channels();
	progress = Mat::zeros(img.size(), CV_8U);
	progress.setTo(255);
	int ncols = img.rows * img.cols;
	for (int i = 0; i < ncols; i++) {
		if (img.at<uchar>(i) == 0) {
			progress.at<uchar>(i) = prog;
		}
	}
}

double Gesture::average(int colors, int pixels) {
	return colors / pixels;
}

long Gesture::variance(const Mat& img, double avg, int ct) {
	long sum = 0;
	for (int y = 0; y < img.cols * img.rows; y++) {
		sum += square(img.at<uchar>(y) - avg);
	}
	return sum / (long)ct;
}

long Gesture::covariance(const Mat& init, const Mat& merged, double avg_init, double avg_merged, int ct) {
	long sum = 0;
	for (int y = 0; y < init.cols * init.rows; y++) {
		sum += (init.at<uchar>(y) - avg_init) * (merged.at<uchar>(y) - avg_merged);
		
	}
	return sum / ((long)ct - 1);
}

long Gesture::square(long diff) {
	return diff * diff;
}

long Gesture::ssim(long avg_x, long avg_y, long covar, long var_x, long var_y) {
	long tl = (2 * avg_x * avg_y) + C1;
	long tr = (2 * covar) + C2;
	long bl = square(avg_x) + square(avg_y) + C1;
	long br = var_x + var_y + C2;

	return (tl * tr) / (bl * br);
}
