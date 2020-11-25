#include "Gesture.h"

Gesture::Gesture() {}

void Gesture::capture() {
	Mat frame, fgMask, fgBlur, fgThresh, merged_frame;
	Ptr<BackgroundSubtractor> pBackSub;
	pBackSub = createBackgroundSubtractorKNN(false);			// trained NN from OpenCV used to remove everything not moving in the background
	cap.open("gesture_tests/test2.mp4");
	//cap.open(0); 
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
		//update_mhi(fgThresh, merged_frame);

		imshow("mhi", fgThresh);
		// show video frame-by-frame
		//get the input from the keyboard
		if (waitKey(10) >= 0)
            break;
	}
	//imwrite("detected_gesture/swipe_left.png", merged_frame);
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
		dst = add(after_mod, progress_set, dst);	// after_mod is the all black img, dst should be the merged
		similarity = Similarity::ssim(init_frame, dst);
		if (abs(similarity - prev_sim) > THRESH1) {
			prog += 8;
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

Mat Gesture::add(Mat& modded, Mat& progress, Mat& merged) {
	Mat dst = Mat::zeros(modded.size(), CV_8U);
	dst.setTo(255);
	uchar p, o, op;
	for (int i = 0; i < modded.cols * modded.rows; i++) {
		p = modded.at<uchar>(i);
		o = progress.at<uchar>(i);
		op = merged.at<uchar>(i);
		if (o != 255) {
			dst.at<uchar>(i) = o;
		}
		if (op != 255) {
			dst.at<uchar>(i) = op;
		}
	}
	return dst;
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