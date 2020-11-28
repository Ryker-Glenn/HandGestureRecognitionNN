#include "Gesture.h"

Gesture::Gesture() {}

void Gesture::capture() {
	VideoCapture cap;
	Mat frame, fgMask, fgBlur, fgThresh, merged_frame;
	Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();
	cap.open("gesture_tests/b.mp4"); 
	//cap.open(0);
	while (true) {
		cap >> frame;
		if (frame.empty()) { break; }

		// Applies background subtractor to the captured frame
		pBackSub->apply(frame, fgMask);

		// apply blur effect to the darkened image
		for (int i = 1; i < MAX_KERNEL_LENGTH; i += 2)
			medianBlur(fgMask, fgBlur, i);						

		// invert the colors of the image
		threshold(fgBlur, fgThresh, 0, MAX_COLOR, THRESH_BINARY_INV | THRESH_OTSU);

		// update the motion history image by merging each frame that is captured
		update_mhi(fgThresh, merged_frame);

		// shows the mhi being updated, won't really need it after testing
		imshow("mhi", merged_frame);
		if (waitKey(10) >= 0)
            break;
	}
	// writes the image to a file just in case its needed later
	//imwrite("detected_gesture/swipe_left.png", merged_frame);
}

void Gesture::update_mhi(Mat& img, Mat& dst) {
	if (dst.size() != img.size()) {
		dst = Mat::zeros(img.size(), CV_8U);	// The code will reach here if it's the first frame
		dst.setTo(255);							// and the mhi hasn't been created yet
	}
	else {
		Mat after_mod, init_frame(img);
		update_progress(img);
		dst = add(img, dst);	
		double similarity = ssim(init_frame, dst);
		if (abs(similarity - prev_sim) > THRESH1)
			prog += 8;

		prev_sim = similarity;
	}
}

/* adds pixels of moving objects to the motion history image		*
 *	by using a bitwise and operation. If a pixel exists in the		*
 *	current frame and not the merge frame, the pixel will be added	*
 *	to the mhi														*/
Mat Gesture::add(Mat& current, Mat& merged) {
	Mat to;
	bitwise_and(current, merged, to);
	return to;
}

/* Lightens pixels of moving objects in the current frame *
 * so that they may be added to the motion history image  */
void Gesture::update_progress(Mat& img) {
	for (int i = 0; i < img.rows * img.cols; i++) 
			img.at<uchar>(i) = (img.at<uchar>(i) == 0) ? prog : img.at<uchar>(i);
}

double Gesture::avg(int px, int count) { return (double)(px / count); }
double Gesture::sq(double to_sq)	   {	    return to_sq * to_sq; }

double Gesture::ssim(const Mat& img, const Mat& merged) {
	// used to get total numerical count of pixels to find average of pixel variance
	int px_init = 0, px_merged = 0, total = img.cols * img.rows;
	double similarity;
	for (int y = 0; y < total; y++) {
		px_init += img.at<uchar>(y);
		px_merged += merged.at<uchar>(y);
	}

	return calc_variances_similarity(img, merged, avg(px_init, total), avg(px_merged, total));
}

double Gesture::calc_variances_similarity(const Mat& img, const Mat& merged, double avx, double avy) {
	long sum_vx = 0, sum_vy = 0, sum_cv = 0;
	int n_ct = img.cols * img.rows;
	for (int y = 0; y < n_ct; y++) {
		sum_vx += sq(img.at<uchar>(y) - avx);
		sum_vy += sq(merged.at<uchar>(y) - avy);
		sum_cv += (img.at<uchar>(y) - avx) * (merged.at<uchar>(y) - avy);
	}

	double tl = (2 * avx * avy) + C1;						// tl and bl are top/bottom left of structural similarity eqxn
	double bl = sq(avx) + sq(avy) + C1;						// which calculates luminance
	double tr = (2 * (sum_cv / n_ct)) + C2;					// while top/bottom right calculate contrast
	double br = (sum_vx / n_ct) + (sum_vy / n_ct) + C2;		// this equation allows calculation of structural simularity

	return (tl * tr) / (bl * br);
}