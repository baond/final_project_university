#include <opencv2/opencv.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\ml\ml.hpp>
using namespace std;
using namespace cv;

//function impliment detection human
//img - origin image
//roi - region interested
//svm - CvSVM detector was trained by using set of HOG descriptor of image
//stride - indicate the decrement coordinate of detect window
//wMinSize - interate detection window 
//scale - ratio of decrement size of detection window after a loop
//wMaxSize - maximum size of detection window
vector<Rect> detect_hog(Mat img, Rect roi, CvSVM svm, Size stride, int wMinSize, float scale, int wMaxSize);


/*
function implement detect human in a region interested
parameter:
	-org: origin image
	-roi: rectagel what indicate the region interested
	-d: HOGDescriptor
	-iterate: number of loop
	-ratio: ratio of decrement of size in each loop
*/
void detectRoiHog(Mat& org, Rect& roi, HOGDescriptor& d, int interate, float ratio,vector<Rect>& detect);

/*Load HOG detector from xml file*/
void loadHOGDescriptor(HOGDescriptor& hog, char* xml);

void detectHOG(Mat& org, Rect& roi, char* svmXml, int minSize, float ratio, int maxSize, Size stride);

void detectRoiHog1(Mat& org, Rect& roi, HOGDescriptor& d, int interate, float ratio, vector<Rect>& detect, vector<Rect>& motion_history);