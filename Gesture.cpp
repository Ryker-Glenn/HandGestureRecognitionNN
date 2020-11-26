#include "Gesture.h"

Gesture::Gesture() {}

void Gesture::capture() {
	Mat frame, fgMask, fgBlur, fgThresh, merged_frame;
	Ptr<BackgroundSubtractor> pBackSub;
	pBackSub = createBackgroundSubtractorMOG2();
	cap.open("gesture_tests/b.mp4"); 
	//cap.open(0);
	while (true) {
		cap >> frame;
		if (frame.empty()) { break; }
		pBackSub->apply(frame, fgMask);
		for (int i = 1; i < MAX_KERNEL_LENGTH; i += 2)
			medianBlur(fgMask, fgBlur, i);						

		threshold(fgBlur, fgThresh, 0, MAX_COLOR, THRESH_BINARY_INV | THRESH_OTSU);
		update_mhi(fgThresh, merged_frame);
		imshow("mhi", fgThresh);
		if (waitKey(10) >= 0)
            break;
	}
	//imwrite("detected_gesture/swipe_left.png", merged_frame);
}

void Gesture::update_mhi(Mat& img, Mat& dst) {
	if (dst.size() != img.size()) {
		dst = img.clone();
	}
	else {
		Mat after_mod, init_frame(img);
		double similarity;
		update_progress(img);
		after_mod = mod(img);
		add(after_mod, dst);	
		similarity = ssim(init_frame, dst);
		if (abs(similarity - prev_sim) > THRESH1)
			prog += 5;

		prev_sim = similarity;
	}
}

Mat Gesture::mod(const Mat& img) {
	Mat after(img);
	for (int y = 0; y < img.cols * img.rows; y++) 
		after.at<uchar>(y) = img.at<uchar>(y) % MAX_COLOR;

	return after;
}

void Gesture::add(Mat& modded, Mat& merged) {
	for (int i = 0; i < modded.cols * modded.rows; i++) {
		if (merged.at<uchar>(i) != modded.at<uchar>(i))
			merged.at<uchar>(i) = modded.at<uchar>(i);
	}
}

void Gesture::update_progress(Mat& img) {
	for (int i = 0; i < img.rows * img.cols; i++) {
			img.at<uchar>(i) = (img.at<uchar>(i) == 0) ? prog : img.at<uchar>(i);
	}
}

double Gesture::avg(int px, int count) { return (double)(px / count); }
double Gesture::sq(double to_sq) { return to_sq * to_sq; }

double Gesture::variance(const Mat& img, double avg, int total) {
	long sum = 0;
	for (int y = 0; y < img.cols * img.rows; y++) 
		sum += sq(img.at<uchar>(y) - avg);
	
	return sum / total;
}

double Gesture::covariance(const Mat& init, const Mat& merged, double avg_init, double avg_merged, int ct) {
	long sum = 0;
	for (int y = 0; y < init.cols * init.rows; y++) 
		sum += (init.at<uchar>(y) - avg_init) * (merged.at<uchar>(y) - avg_merged);
	
	return sum / ct;
}

double Gesture::ssim(const Mat& img, const Mat& merged) {
	int px_init = 0, px_merged = 0, total = 0;
	double avg_img, avg_merged, v_img, v_merged, covar;
	for (int y = 0; y < img.cols * img.rows; y++) {
		total++;
		px_init += img.at<uchar>(y);
		px_merged += merged.at<uchar>(y);
	}
	avg_img = avg(px_init, total);
	avg_merged = avg(px_merged, total);
	v_img = variance(img, avg_img, total);
	v_merged = variance(merged, avg_merged, total);
	covar = covariance(img, merged, avg_img, avg_merged, total);
	return calc_ssim(avg_img, avg_merged, v_img, v_merged, covar);
}

double Gesture::calc_ssim(double avx, double avy, double vx, double vy, double cv) {
	double tl = (2 * avx * avy) + C1;	// tl and bl are top/bottom left of structural similarity eqxn
	double bl = sq(avx) + sq(avy) + C1;	// which calculates luminance
	double tr = (2 * cv) + C2;			// while top/bottom right calculate contrast
	double br = vx + vy + C2;			// this equation allows calculation of structural simularity

	return (tl * tr) / (bl * br);
}