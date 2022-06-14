#pragma once
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "ArucoDetector.h"

using namespace std;
using namespace cv;

Mat CAM_MTX = (Mat_<float>(3, 3) << 1000, 0.0, 500, 0.0, 1000, 500, 0.0, 0.0, 1.0);
Mat CAM_DIST = (Mat_<float>(1, 4) << 0, 0, 0, 0);

namespace Utils {
	void drawGrid(Mat& img, int rows, int cols, Scalar color = Scalar(0, 0, 255)) {
		int cellW = img.cols / cols;
		int cellH = img.rows / rows;

		for (int i = 1; i < rows; ++i) {
			double y = i * cellH;
			line(img, Point(0, y), Point(img.cols, y), color, 1, LINE_AA);
		}

		for (int i = 1; i < cols; ++i) {
			double x = i * cellW;
			line(img, Point(x, 0), Point(x, img.rows), color, 1, LINE_AA);
		}
	}

	void drawContourFloat(Mat& img, vector<Point2f> cnt, Scalar color = Scalar(0, 0, 255)) {
		for (int i = 0; i < cnt.size(); ++i) {
			Point2f from = cnt[i];
			Point2f to = cnt[(i + 1) % cnt.size()];
			line(img, from, to, color, 1, LINE_AA);
		}
	}

	void drawContoursFloat(Mat& img, vector<vector<Point2f>> contours, Scalar color = Scalar(0, 0, 255)) {
		for (vector<Point2f>& cnt : contours)
		{
			drawContourFloat(img, cnt, color);
		}
	}

	void drawArucos(Mat& img, vector<ArucoResult> ars, Scalar color = Scalar(0, 0, 255)) {
		for (ArucoResult& ar : ars)
		{
			drawContourFloat(img, ar.corners);
			putText(img, to_string(ar.index), ar.corners[0], FONT_HERSHEY_SIMPLEX, 1, color);
		}
	}

	void drawAxisWithPose(Mat& img, vector<ArucoResult> ars, ArucoDict dict) {

		vector<Point3f> axis = {Point3f(0.0, 0.0, 0.0), Point3f(25.0, 0.0, 0.0) , Point3f(0.0, 25.0, 0.0), Point3f(0.0, 0.0, -25.0) };
		vector<Point3f> square = { Point3f(0.0, 30.0, 0.0), Point3f(50.0, 30.0, 0.0) , Point3f(50.0, 80.0, 0.0), Point3f(0.0, 80.0, 0.0) };

		for (ArucoResult& ar : ars)
		{
			Mat rvec;
			Mat tvec;

			solvePnP(dict.worldLoc[ar.index], ar.corners, CAM_MTX, CAM_DIST, rvec, tvec);

			vector<Point2f> imgpts;

			// draw axis on marker
			projectPoints(axis, rvec, tvec, CAM_MTX, CAM_DIST, imgpts);
			line(img, imgpts[0], imgpts[1], Scalar(0, 0, 255), 1, LINE_AA); // X-axis
			line(img, imgpts[0], imgpts[2], Scalar(0, 255, 0), 1, LINE_AA); // Y-axis
			line(img, imgpts[0], imgpts[3], Scalar(255, 0, 0), 1, LINE_AA); // Z-axis

			// Draw square below marker
			projectPoints(square, rvec, tvec, CAM_MTX, CAM_DIST, imgpts);
			line(img, imgpts[0], imgpts[1], Scalar(0, 0, 255), 1, LINE_AA);
			line(img, imgpts[1], imgpts[2], Scalar(0, 0, 255), 1, LINE_AA);
			line(img, imgpts[2], imgpts[3], Scalar(0, 0, 255), 1, LINE_AA);
			line(img, imgpts[3], imgpts[0], Scalar(0, 0, 255), 1, LINE_AA);
		}
	}

	void drawCrop(Mat& img, vector<ArucoResult> ars, ArucoDict dict) {

		if(ars.size() <= 2) return;
		
        // draw axis on marker
		Point2f p1 = Point2f((ars[0].corners[0].x + ars[0].corners[1].x + ars[0].corners[2].x + ars[0].corners[3].x) / 4, (ars[0].corners[0].y + ars[0].corners[1].y + ars[0].corners[2].y + ars[0].corners[3].y) / 4);
		Point2f p2 = Point2f((ars[2].corners[0].x + ars[2].corners[1].x + ars[2].corners[2].x + ars[2].corners[3].x) / 4, (ars[2].corners[0].y + ars[2].corners[1].y + ars[2].corners[2].y + ars[2].corners[3].y) / 4);
		rectangle(img, p1, p2, Scalar(255, 0, 255), 2);

		// line(img, ars[0].corners[0], ars[1].corners[0], Scalar(0, 0, 255), 1, LINE_AA);
		// line(img, ars[1].corners[0], ars[2].corners[0], Scalar(0, 0, 255), 1, LINE_AA);
		// line(img, ars[2].corners[0], ars[0].corners[0], Scalar(0, 0, 255), 1, LINE_AA);

		// vector<Point3f> axis = {Point3f(0.0, 0.0, 0.0), Point3f(25.0, 0.0, 0.0) , Point3f(0.0, 25.0, 0.0), Point3f(0.0, 0.0, -25.0) };
		// vector<Point3f> square = { Point3f(0.0, 30.0, 0.0), Point3f(50.0, 30.0, 0.0) , Point3f(50.0, 80.0, 0.0), Point3f(0.0, 80.0, 0.0) };

		// for (ArucoResult& ar : ars)
		// {
		// 	Mat rvec;
		// 	Mat tvec;

		// 	solvePnP(dict.worldLoc[ar.index], ar.corners, CAM_MTX, CAM_DIST, rvec, tvec);

		// 	vector<Point2f> imgpts;

		// 	// draw axis on marker
		// 	projectPoints(axis, rvec, tvec, CAM_MTX, CAM_DIST, imgpts);
		// 	// line(img, imgpts[0], imgpts[1], Scalar(0, 0, 255), 1, LINE_AA); // X-axis
		// 	// line(img, imgpts[0], imgpts[2], Scalar(0, 255, 0), 1, LINE_AA); // Y-axis
		// 	// line(img, imgpts[0], imgpts[3], Scalar(255, 0, 0), 1, LINE_AA); // Z-axis

		// 	// Draw square below marker
		// 	projectPoints(square, rvec, tvec, CAM_MTX, CAM_DIST, imgpts);
		// 	line(img, imgpts[0], imgpts[1], Scalar(0, 0, 255), 1, LINE_AA);
		// 	line(img, imgpts[1], imgpts[2], Scalar(0, 0, 255), 1, LINE_AA);
		// 	line(img, imgpts[2], imgpts[3], Scalar(0, 0, 255), 1, LINE_AA);
		// 	line(img, imgpts[3], imgpts[0], Scalar(0, 0, 255), 1, LINE_AA);
		// }
	}

	void drawStatusText(Mat& img, string text,  int x = 10, int y = 30, Scalar color = Scalar(0, 0, 255)) {
		putText(img, text, Point(x,y), FONT_HERSHEY_SIMPLEX, 0.5, color);
	}
	void drawSquareCenterText(Mat& img, vector<Point2f> cnt, Scalar color = Scalar(0, 0, 255)) {
		if(cnt.size() == 4) {
			Point2f center = Point2f((cnt[0].x + cnt[1].x + cnt[2].x + cnt[3].x) / 4, (cnt[0].y + cnt[1].y + cnt[2].y + cnt[3].y) / 4);
			putText(img, "(" + to_string(center.x) + ", " + to_string(center.y) + ")", center, FONT_HERSHEY_SIMPLEX, 0.5, color);
		}
	}
}
