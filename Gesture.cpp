#include "Gesture.h"

Gesture::Gesture() {}

void Gesture::capture() {
	Mat frame, fgMask, fgBlur, fgThresh;
	Ptr<BackgroundSubtractor> pBackSub;
	pBackSub = createBackgroundSubtractorMOG2();			// trained NN from OpenCV used to remove everything not moving in the background
	cap.open(0);												// Open webcam
	while (true) {
		cap >> frame;
		if (frame.empty()) { break; }
		pBackSub->apply(frame, fgMask);							// apply background remover to current video
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		for (int i = 1; i < MAX_KERNEL_LENGTH; i += 2)
			medianBlur(fgMask, fgBlur, i);						// For the kernel size, apply opencv's median blur function to frame with removed
																// background
		threshold(fgMask, fgThresh, 0, MAX_BV, BI);				// apply binary inverted threshold to the blurred frame
		imshow("BI Blur FG Mask", fgThresh);
		// show video frame-by-frame
		//get the input from the keyboard
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;
	}

}

Mat Gesture::modulo(Mat img, int mod) {
	Mat after_mod = img.clone();
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			after_mod.at<uchar>(y, x) = after_mod.at<uchar>(y, x) % mod;
		}
	}
	return after_mod;
}

void Gesture::update_mhi(Mat img) {
	Mat after_mod, mf;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			if (img.at<uchar>(y, x) == 0) {
				img.at<uchar>(y, x) = progress;
			}
		}
	}

	after_mod = modulo(img, 255);
	mf = add(after_mod, img);
}

Mat Gesture::add(Mat after, Mat merged) {
	addWeighted(after, 0.5, merged, 0.5, 0.0, merged);
	return merged;
}
