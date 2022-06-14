#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Utils.h"
#include "ArucoDetector.h"

using namespace std;
using namespace cv;

int main()
{
	Mat marker = imread("../imgs/marker_black.png");
	ArucoDetector detector(marker, 36);

	VideoCapture cap(1);
	while (true) {
		Mat frame;
		cap >> frame;

		Mat gray;
		cvtColor(frame, gray, COLOR_BGRA2GRAY);
		vector<ArucoResult> ars = detector.detectArucos(gray, 1);
		// Utils::drawArucos(frame, ars);
		Utils::drawCrop(frame, ars, detector.m_dict);
		Utils::drawStatusText(frame, "Found " + to_string(ars.size()) + " squares", 10, 30);
		imshow("frame", frame);
		int k = waitKey(50);
		if (k >= 0) {
			break;
		}
	}
    return 0;
}