#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>


#define C1		6.5025		// Calculated as (k_1 * L)^2 where k_1 = 0.01 and L is 2^8 - 1
#define C2		58.5225		// Calculated as (k_2 * L)^2 where k_2 = 0.03 and L is 2^8 - 1
#define C3		C2 / 2

using namespace std;
using namespace cv;

static class Similarity {
public:
	static double average(int p_count, int total) { return p_count / total; }
	
	static long square(double i) { return (long)(i * i); }

	static long variance(const Mat& img, double avg, int total) {
			long sum = 0;
			for (int y = 0; y < img.cols * img.rows; y++) {
				sum += square(img.at<uchar>(y) - avg);
			}
			return sum / (long)total;
	}

	static long luminance(double avg_x, double avg_y) {
		return ((2 * avg_x * avg_y) + C1) / (square(avg_x) + square(avg_y) + C1);
	}

	static long contrast(long v_img, long v_merged) {
		return ((2 * v_img * v_merged) + C2) / (square(v_img) + square(v_merged) + C2);
	}

	static long structure(long covar, long var_init, long var_merged) {
		return (covar + C3) / ((var_init * var_merged) + C3);
	}

	static long covariance(const Mat& init, const Mat& merged, double avg_init, double avg_merged, int ct) {
		long sum = 0;
		for (int y = 0; y < init.cols * init.rows; y++) {
			sum += (init.at<uchar>(y) - avg_init) * (merged.at<uchar>(y) - avg_merged);
		}
		return sum / ((long)ct - 1);
	}


	static long ssim(const Mat& img, const Mat& merged) {
		int px_init = 0, px_merged = 0, total = 0;
		double avg_img, avg_merged;
		long lumin, contr, structs, v_img, v_merged, covar;
		for (int y = 0; y < img.cols * img.rows; y++) {
			total++;
			px_init += img.at<uchar>(y);
			px_merged += merged.at<uchar>(y);
		}
		avg_img = average(px_init, total);
		avg_merged = average(px_merged, total);
		v_img = variance(img, avg_img, total);
		v_merged = variance(merged, avg_merged, total);
		covar = covariance(img, merged, avg_img, avg_merged, total);
		
		return luminance(avg_img, avg_merged) * contrast(v_img, v_merged) * structure(covar, v_img, v_merged);
	}
};

