#include "detect_HOG.h"

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
vector<Rect> detect_hog(Mat img, Rect roi, CvSVM svm, Size stride, int wMinSize, float scale, int wMaxSize){
	HOGDescriptor d(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	vector<Rect> detection;
	int size = wMinSize;
	Point position = Point(roi.x, roi.y);
	Rect detectWin = Rect(position, Size(size, 2 * size));
	while ((position.y + size * 2 < roi.height)){
		while (size < wMaxSize && (position.x + size < roi.width) && (position.y + size * 2 < roi.height)){
			Mat detectMat = img(detectWin);
			resize(detectMat, detectMat, Size(64, 128));
			cvtColor(detectMat, detectMat, CV_BGR2GRAY);

			//compute HOG feature
			vector< float> descriptorsValues;
			vector< Point> locations;
			d.compute(detectMat, descriptorsValues, Size(0, 0), Size(0, 0), locations);

			//vector to Mat  
			Mat fm = Mat(descriptorsValues);

			int result = svm.predict(fm);
			if (result == 1){
				detection.push_back(detectWin);
			}

			size = size * scale;
		}
		size = wMinSize;
		position = Point(position.x + stride.width, position.y);
		if ((position.x + size > roi.width)){
			position = Point(0, position.y + stride.height);
		}
	}

	return detection;
}

int normalizeImage(Mat &img, Mat& nIm, Size sz){
	Mat normImg = img;
	int scale = 0;
	while (normImg.rows < sz.height || normImg.cols < sz.width){
		Mat temp;
		pyrUp(normImg, temp, Size(normImg.cols * 2, normImg.rows * 2));
		normImg = temp;
		nIm = temp;
		scale++;
	}
	return scale;
}

//sharpen an image
void unsharpMask(Mat& im)
{
	Mat tmp;
	GaussianBlur(im, tmp, Size(5, 5), 5);
	addWeighted(im, 1.5, tmp, -0.5, 0, im);
}

void drawRect1(vector<Rect> ds, Mat im, float scale, Rect org, vector<Rect> & motion_history){
	int x, y, h, w;
	for (int i = 0; i < ds.size(); i++){
		x = org.x + (int)ds[i].x / scale;
		y = org.y + (int)ds[i].y / scale;
		h = ds[i].height / scale;
		w = ds[i].width / scale;

		Rect obj = Rect(x, y, w, h);
		motion_history.push_back(obj);
		Scalar color = Scalar(10, 10, 250);
		rectangle(im, obj, color);
	}
}

void drawRect2(vector<Rect> ds, Mat im, float scale, Rect org){
	int x, y, h, w;
	for (int i = 0; i < ds.size(); i++){
		x = org.x + (int)ds[i].x / scale;
		y = org.y + (int)ds[i].y / scale;
		h = ds[i].height / scale;
		w = ds[i].width / scale;

		Rect obj = Rect(x, y, w, h);

		Scalar color = Scalar(10, 10, 250);
		rectangle(im, obj, color);
	}
}

void detectRoiHog(Mat& org, Rect& roi, HOGDescriptor& d, int interate, float ratio, vector<Rect>& detect){
	Mat roiImg = org(roi);
	int s = normalizeImage(roiImg, roiImg, Size(64, 128));
	float scale = pow(2, s);
	vector<Rect> found;
	for (int i = 0; i < interate; i++){
		d.detectMultiScale(roiImg, found, 0.8, Size(8, 8), Size(0, 0), 1.01, 10);
		if (found.size() != 0){
			drawRect2(found, org, scale, roi);
			detect = found;
			break;
		}
		else{
			scale = scale * ratio;
			resize(roiImg, roiImg, Size(roiImg.cols*ratio, roiImg.rows*ratio));
		}

	}
}

void detectRoiHog1(Mat& org, Rect& roi, HOGDescriptor& d, int interate, float ratio, vector<Rect>& detect, vector<Rect>& motion_history){
	Mat roiImg = org(roi);
	int s = normalizeImage(roiImg, roiImg, Size(64, 128));
	float scale = pow(2, s);
	vector<Rect> found;
	for (int i = 0; i < interate; i++){
		d.detectMultiScale(roiImg, found, 0.8, Size(8, 8), Size(0, 0), 1.01, 10);
		if (found.size() != 0){
			drawRect1(found, org, scale, roi, motion_history);
			cout << "size detect: " << found.size() << '\n';
			detect = found;
			break;
		}
		else{
			scale = scale * ratio;
			resize(roiImg, roiImg, Size(roiImg.cols*ratio, roiImg.rows*ratio));
		}

	}
}

void detectHOG(Mat& org, Rect& roi, char* svmXml, int minSize, float ratio, int maxSize, Size stride){
	cout << roi.x << " " << roi.y << ' ' << roi.width << ' ' << roi.height << '\n';
	Mat roiImg = org(roi);
	vector<Rect> detection;
	HOGDescriptor d;
	CvSVM svm;
	svm.load(svmXml);
	int wRange = minSize;
	if (maxSize > roi.width || maxSize * 2 > roi.height || maxSize < minSize)
		return;
	while (wRange <= maxSize)
	{
		cout << "in\n";
		cout << wRange << '\n';
		Size wSize = Size(wRange, 2 * wRange);
		for (int i = 0; i + stride.height + 2 * wRange < roi.height; i += stride.height){
			for (int j = 0; j + stride.width + wRange < roi.width; j += stride.width){
				Rect windows(j, i, wSize.width, wSize.height);
				cout << "windetect " << i << ' ' << j << ' ' << windows.width << ' ' << windows.height << '\n';
				Mat winDetect = roiImg(windows);
				resize(winDetect, winDetect, Size(64, 128));
				cvtColor(winDetect, winDetect, CV_RGB2GRAY);
				vector< float> descriptorsValues;
				vector< Point> locations;
				d.compute(winDetect, descriptorsValues, Size(0, 0), Size(0, 0), locations);
				Mat fm = Mat(descriptorsValues);
				int result = svm.predict(fm);
				if (result == 1){
					detection.push_back(windows);
				}

				descriptorsValues.clear();
				locations.clear();
			}
		}

		wRange = wRange * ratio;
	}

	if (detection.size() != 0){
		drawRect2(detection, org, 1, roi);
	}
}

void loadHOGDescriptor(HOGDescriptor& hog, char* xml){
	CvSVM cSvm;
	cSvm.load(xml);

	//getPrimalForm
	vector<float> support_vector;
	int sv_count = cSvm.get_support_vector_count();
	const CvSVMDecisionFunc* df = cSvm.getDecisionFunc();
	const double* alphas = df[0].alpha;
	double rho = df[0].rho;
	int var_count = cSvm.get_var_count();

	support_vector.resize(var_count, 0);
	for (unsigned int r = 0; r < (unsigned)sv_count; r++)
	{
		float myalpha = alphas[r];
		const float* v = cSvm.get_support_vector(r);
		for (int j = 0; j < var_count; j++, v++)
		{
			support_vector[j] += (-myalpha) * (*v);
		}
	}
	support_vector.push_back(rho);
	hog.setSVMDetector(support_vector);
}