#include "connected_componet.h"

using namespace cv;
using namespace std;

void labelingCC(Mat& im, Mat& org, int**& LBMat, vector<Rect>& motion, int thres){
	int numOfMersOb = 0;
	Point *Position;
	int lbSetSize = 0;
	int *lbsize;// size of each label 
	int sizex, sizey;
	sizex = org.rows;
	sizey = org.cols;

	LBMat = new int*[sizex];
	for (int a = 0; a < sizex; a++){
		LBMat[a] = new int[sizey];
	}

	for (int i = 0; i < sizex; i++){
		for (int j = 0; j < sizey; j++)
			LBMat[i][j] = 0;
	}

	int *id = new int[sizex*sizey];
	int *size = new int[sizex*sizey];
	for (int i = 0; i < sizex*sizey; i++){
		id[i] = i;
		size[i] = 0;
	}
	int maxLabel = 0;



	for (int i = 0; i < im.rows; i++){
		for (int j = 0; j < im.cols; j++){
			if (im.at<uchar>(i, j) != 0){
				if (i == 0){
					if (j == 0 || im.at<uchar>(i, j - 1) == 0){
						LBMat[i][j] = ++maxLabel;
						size[maxLabel]++;
					}
					else{
						LBMat[i][j] = LBMat[i][j - 1];
						size[LBMat[i][j]]++;
					}
				}
				else{
					if (im.at<uchar>(i - 1, j) == 0){
						if (j == 0 || im.at<uchar>(i, j - 1) == 0){
							LBMat[i][j] = ++maxLabel;
							size[maxLabel]++;
						}
						else{
							LBMat[i][j] = LBMat[i][j - 1];
							size[LBMat[i][j]]++;
						}
					}
					else{
						if (j == 0 || im.at<uchar>(i, j - 1) == 0){
							LBMat[i][j] = LBMat[i - 1][j];
							size[LBMat[i][j]]++;
						}
						else{
							LBMat[i][j] = MIN(LBMat[i - 1][j], LBMat[i][j - 1]);
							size[LBMat[i][j]]++;
							if (LBMat[i][j] == LBMat[i - 1][j]){
								id[LBMat[i][j - 1]] = LBMat[i][j];
							}
							else
								id[LBMat[i - 1][j]] = LBMat[i][j];
						}
					}
				}
			}
		}
	}


	// pass 2: update label
	for (int i = 0; i < im.rows; i++){
		for (int j = 0; j < im.cols; j++){
			size[LBMat[i][j]]--;
			int a = LBMat[i][j];
			bool temp = false;
			while (id[LBMat[i][j]] != LBMat[i][j]){
				LBMat[i][j] = id[LBMat[i][j]];
				temp = true;
			}
			size[LBMat[i][j]]++;
		}
	}


	// check number of label
	for (int i = 0; i < maxLabel; i++){
		if (size[i] >= 10){
			lbSetSize++;
		}
	}

	int *lb;
	lbsize = new int[lbSetSize];
	lb = new int[lbSetSize];
	Position = new Point[lbSetSize];



	lbSetSize = 0;
	for (int i = 0; i < maxLabel; i++){
		if (size[i] >= 10){
			lb[lbSetSize] = i;
			lbsize[lbSetSize] = size[i];
			lbSetSize++;
		}
	}
	delete[] size;


	int len = 0;
	vector<Rect> rec;
	for (int i = 0; i < lbSetSize; i++){
		int *blobx, *bloby, cnt = 0;
		blobx = new int[sizex*sizey];
		bloby = new int[sizex*sizey];
		for (int j = 0; j < sizex; j++){
			for (int k = 0; k < sizey; k++){
				if (LBMat[j][k] == lb[i]){
					blobx[cnt] = j;
					bloby[cnt] = k;
					cnt++;
				}
			}
		}
		std::sort(blobx, blobx + cnt);
		sort(bloby, bloby + cnt);
		int maxx, maxy, minx, miny;
		minx = bloby[0];
		miny = blobx[0];
		maxx = bloby[cnt - 1];
		maxy = blobx[cnt - 1];
		Position[i] = Point((int)((maxx + minx) / 2), (int)(maxy / 2 + miny / 2));
		Rect obj(Point(minx, miny), Size(maxx - minx, maxy - miny));
		rec.push_back(obj);

		len++;

		delete[] blobx;
		delete[] bloby;
	}

	vector<int>* merg = new vector<int>[len];
	int c = 0;
	for (int i = 0; i < len; i++){
		bool check = false;
		for (int k = 0; k < len; k++){
			if (!merg[k].empty() && find(merg[k].begin(), merg[k].end(), i) != merg[k].end())
				check = true;
		}
		if (!check){
			merg[c].push_back(i);
			for (int j = 0; j < len; j++){
				if (!(find(merg[c].begin(), merg[c].end(), j) != merg[c].end())){
					Rect a = rec.at(j);
					for (int k = 0; k < merg[c].size(); k++){
						Rect b = rec.at(merg[c].at(k));
						int rangex = abs(Position[merg[c].at(k)].x - Position[j].x) - a.width / 2 - b.width / 2;
						int rangey = abs(Position[merg[c].at(k)].y - Position[j].y) - a.height / 2 - b.height / 2;
						if (rangex < thres && rangey < thres){
							merg[c].push_back(j);
							j = -1;
						}
					}
				}
			}

			c++;
		}
	}


	for (int i = 0; i < len; i++){
		if (!merg[i].empty())
			numOfMersOb++;
	}

	int* sizeOfMersOb = new int[numOfMersOb];

	for (int i = 0; i < numOfMersOb; i++)
		sizeOfMersOb[i] = 0;

	numOfMersOb = 0;
	for (int i = 0; i < len; i++){
		int cc = 0;
		int *left, *right, *top, *bottom;
		left = new int[len];
		right = new int[len];
		top = new int[len];
		bottom = new int[len];
		if (!merg[i].empty()){
			for (int j = 0; j < merg[i].size(); j++){
				top[cc] = Position[merg[i].at(j)].x - rec.at(merg[i].at(j)).width / 2;
				left[cc] = Position[merg[i].at(j)].y - rec.at(merg[i].at(j)).height / 2;
				bottom[cc] = Position[merg[i].at(j)].x + rec.at(merg[i].at(j)).width / 2;
				right[cc] = Position[merg[i].at(j)].y + rec.at(merg[i].at(j)).height / 2;
				sizeOfMersOb[numOfMersOb] += lbsize[merg[i].at(j)];
				cc++;
			}

			sort(top, top + cc);
			sort(left, left + cc);
			sort(right, right + cc);
			sort(bottom, bottom + cc);

			Rect obj(Point(top[0], left[0]), Size(bottom[cc - 1] - top[0], right[cc - 1] - left[0]));
			motion.push_back(obj);
			
			numOfMersOb++;
			delete[] left;
			delete[] right;
			delete[] bottom;
			delete[] top;
		}

	}
}