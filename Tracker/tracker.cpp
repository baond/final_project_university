#include <opencv\cv.h>
#include <opencv\highgui.h>
#include "detect_HOG.h"
#include "connected_componet.h"
#include <fstream>
#include <thread>
#include <opencv2\video\background_segm.hpp>


using namespace std;
using namespace cv;


//our sensitivity value to be used in the absdiff() function
const static int SENSITIVITY_VALUE = 20;
//size of blur used to smooth the intensity image output from absdiff() function
const static int BLUR_SIZE = 10;


int **LBMat;
vector<Rect> human;
Ptr<BackgroundSubtractorMOG2> bgSubtractor;

void taskDetect(Mat org, Rect motion, char* xml){
	vector<Rect> detect;
	HOGDescriptor hog;
	loadHOGDescriptor(hog, xml);
	if (motion.width*motion.height > 25){
		detectRoiHog(org, motion, hog, 5, 1.2, detect);
	}
}

bool isOverlap(Rect a, Rect b){
	Size diff = Size(abs(a.x + a.width / 2 - b.x - b.width / 2), abs(a.y + a.height / 2 - b.y - b.height / 2));
	if (diff.width < (a.width + b.width) / 2 && diff.height < (a.height + b.height) / 2){
		return true;
	}
	else{
		return false;
	}

}

int main(){
	//some boolean variables for added functionality
	bool objectDetected = false;
	//these two can be toggled by pressing 'd' or 't'
	bool debugMode = false;
	bool trackingEnabled = true;
	//pause and resume code
	bool pause = false;
	//set up the matrices that we will need
	//the two frames we will be comparing
	Mat frame1, frame2;
	//their grayscale images (needed for absdiff() function)
	Mat grayImage1, grayImage2;
	//resulting difference image
	Mat differenceImage;
	//thresholded difference image (for use in findContours() function)
	Mat thresholdImage;
	//mask background subtrack
	Mat mask;
	//video capture object.
	VideoCapture capture;

	//get HOG detector
	HOGDescriptor hog;
	//Load detector from xml file 
	loadHOGDescriptor(hog, "train2.xml");

	vector<Rect> motion_history;



	bgSubtractor = new BackgroundSubtractorMOG2();
	while (1){

		//we can loop the video by re-opening the capture every time the video reaches its last frame

		int count = 0;
		//capture.open("Time_12-34\\View_006\\frame_%04d.jpg");
		capture.open("PETS2009_sample_1.avi");
		//capture.open("TUD-Stadtmitte.mp4");
		//capture.open("walking.avi");
		const clock_t begin_time = clock();
		if (!capture.isOpened()){
			cout << "ERROR ACQUIRING VIDEO FEED\n";
			getchar();
			return -1;
		}


		//check if the video has reach its last frame.
		//we add '-1' because we are reading two frames from the video at a time.
		//if this is not included, we get a memory error!
		while (capture.get(CV_CAP_PROP_POS_FRAMES) < capture.get(CV_CAP_PROP_FRAME_COUNT) - 1){

			//read first frame
			capture.read(frame1);
			count++;
			printf("frame %d\n", count);
			//convert frame1 to gray scale for frame differencing
			resize(frame1, frame1, Size(320, 240), 0, 0, INTER_CUBIC);
			cvtColor(frame1, grayImage1, COLOR_BGR2GRAY);
			//copy second frame
			capture.read(frame2);
			resize(frame2, frame2, Size(320, 240), 0, 0, INTER_CUBIC);
			//convert frame2 to gray scale for frame differencing
			cvtColor(frame2, grayImage2, COLOR_BGR2GRAY);
			//perform frame differencing with the sequential images. This will output an "intensity image"
			//do not confuse this with a threshold image, we will need to perform thresholding afterwards.
			absdiff(grayImage1, grayImage2, differenceImage);
			//threshold intensity image at a given sensitivity value
			threshold(differenceImage, thresholdImage, SENSITIVITY_VALUE, 255, THRESH_BINARY);
			if (debugMode == true){
				//show the difference image and threshold image
				imshow("Difference Image", differenceImage);
				imshow("Threshold Image", thresholdImage);
			}
			else{
				//if not in debug mode, destroy the windows so we don't see them anymore
				destroyWindow("Difference Image");
				destroyWindow("Threshold Image");
			}
			//blur the image to get rid of the noise. This will output an intensity image
			blur(thresholdImage, thresholdImage, cv::Size(BLUR_SIZE, BLUR_SIZE));
			//threshold again to obtain binary image from blur output
			threshold(thresholdImage, thresholdImage, SENSITIVITY_VALUE, 255, THRESH_BINARY);

			bgSubtractor->operator()(frame1, mask);

			if (debugMode == true){
				//show the threshold image after it's been "blurred"

				imshow("Final Threshold Image", thresholdImage);

			}
			else {
				//if not in debug mode, destroy the windows so we don't see them anymore
				destroyWindow("Final Threshold Image");
			}

			//if tracking enabled, search for contours in our thresholded image
			if (trackingEnabled){
				vector<Rect> motion, motion1;
				vector<Rect> detection;
				medianBlur(mask, mask, 5);
				labelingCC(thresholdImage, frame1, LBMat, motion, 5);


				for (int k = 0; k < motion.size(); k++){
					rectangle(frame1, motion[k], Scalar(200, 20, 0));
					}

				//labelingCC(mask, frame1, LBMat, motion1, 5);
				//for (int k = 0; k < motion1.size(); k++){
				//	if (motion1[k].width > 5 && motion1[k].height > 5){
				//		/*motion1[k].x = MAX(motion1[k].x - 5, 0);
				//		motion1[k].y = MAX(motion1[k].y - 5, 0);
				//		motion1[k].width = MIN(motion1[k].width + 10, frame1.cols - motion1[k].x);
				//		motion1[k].height = MIN(motion1[k].height + 10, frame1.rows - motion1[k].y);*/
				//		rectangle(frame1, motion1[k], Scalar(51, 255, 255));
				//	}
				//}
				//Size average = Size(0, 0);
				//if (motion_history.size() != 0){
				//	for (int k = 0; k < motion_history.size(); k++){
				//		average.width += motion_history[k].width;
				//		average.height += motion_history[k].height;
				//	}
				//	average = Size(average.width / motion_history.size(), average.height / motion_history.size());
				//}
				//cout << "average size: " << average << '\n';
				//motion_history.clear();

				//for (int k = 0; k < motion.size(); k++){					
				//	if (motion_history.size() != 0){
				//		if (motion[k].width < average.width / 3 && motion[k].height < average.height / 3){
				//			//detectRoiHog1(frame1, motion[k], hog, 5, 1.2, detect, motion_history);
				//			for (int n = 0; n < motion_history.size(); n++){
				//				if (isOverlap(motion[k], motion_history[n])){
				//					if (motion_history[n].width > motion[k].width && motion_history[n].height > motion[k].height){
				//						motion[k].x = MAX(0,motion[k].x + motion[k].width / 2 - motion_history[n].width/2);
				//						motion[k].y = MAX(0, motion[k].y + motion[k].height / 2 - motion_history[n].height / 2);
				//						motion[k].width = MIN(motion_history[n].width, frame1.cols - motion[k].x);
				//						motion[k].height = MIN(motion_history[n].height, frame1.rows - motion[k].y);
				//					}
				//				}
				//			}
				//		}
				//	}					
				//}

				/*for (int m = 0; m < motion.size(); m++){
					if (motion[m].width * motion[m].height > 4 * average.width * average.height){
						bool check = false;
						for (int n = 0; n < motion1.size(); n++){
							if (motion[m].x <= motion1[n].x && motion[m].y <= motion1[n].y &&
								motion[m].x + motion[m].width >= motion1[n].x + motion1[n].width  &&
								motion[m].y + motion[m].height >= motion1[n].y + motion1[n].height &&
								motion1[n].width * motion1[n].height >= average.width * average.height / 2 &&
								motion1[n].width * motion1[n].height <= 2 * average.width * average.height){
								cout << "true true\n";
								motion1[n].x = MAX(motion1[n].x - 5, 0);
								motion1[n].y = MAX(motion1[n].y - 5, 0);
								motion1[n].width = MIN(motion1[n].width + 10, frame1.cols - motion1[n].x);
								motion1[n].height = MIN(motion1[n].height + 10, frame1.rows - motion1[n].y);
								motion.push_back(motion1[n]);
								check = true;
							}
						}
						if (check)
							motion.erase(motion.begin() + m);
					}
				}*/

				for (int k = 0; k < motion.size(); k++){
					Scalar color = Scalar(0, 0, 0);
					rectangle(frame1, motion[k], color);
					if (motion[k].width > 10 && motion[k].height > 10){
						detectRoiHog1(frame1, motion[k], hog, 5, 1.3, detection, motion_history);
					}
				}


				std::cout << "time elapse: " << float(clock() - begin_time) / CLOCKS_PER_SEC << '\n';

				/*ofstream result;
				result.open("result.txt", ios::app);
				result << "Frame " << count;
				result << " number of detection: " << detection.size() << " ";
				for (int i = 0; i < detection.size(); i++){
				result << "object" << i + 1 << " ";
				result << "(" << detection[i].x << ", " << detection[i].y << ") (" << detection[i].x + detection[i].width << ", " << detection[i].y + detection[i].height << ") ";
				}
				result << "\n";

				result.close();*/

				detection.clear();
				motion.clear();
			}

			//show our captured frame
			imshow("Frame1", frame1);

			//imwrite("img//" + to_string(count)+".jpg", frame1);
			//check to see if a button has been pressed.
			//this 5ms delay is necessary for proper operation of this program
			//if removed, frames will not have enough time to referesh and a blank 
			//image will appear.
			switch (waitKey(5)){

			case 27: //'esc' key has been pressed, exit program.
				return 0;
			case 116: //'t' has been pressed. this will toggle tracking
				trackingEnabled = !trackingEnabled;
				if (trackingEnabled == false) cout << "Tracking disabled." << endl;
				else cout << "Tracking enabled." << endl;
				break;
			case 100: //'d' has been pressed. this will debug mode
				debugMode = !debugMode;
				if (debugMode == false) cout << "Debug mode disabled." << endl;
				else cout << "Debug mode enabled." << endl;
				break;
			case 112: //'p' has been pressed. this will pause/resume the code.
				pause = !pause;
				if (pause == true){
					cout << "Code paused, press 'p' again to resume" << endl;
					while (pause == true){
						//stay in this loop until 
						switch (waitKey()){
							//a switch statement inside a switch statement? Mind blown.
						case 112:
							//change pause back to false
							pause = false;
							cout << "Code Resumed" << endl;
							break;
						}
					}
				}



			}
			delete[] LBMat;
			human.clear();

		}
		//release the capture before re-opening and looping again.
		capture.release();
	}

	return 0;

}