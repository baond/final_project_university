#include <opencv2\opencv.hpp>

using namespace cv;

void labelingCC(Mat& im, Mat& org, int**& LBMat, vector<Rect>& motion, int thres);