#include "opencv2/opencv.hpp"
#include <map>
#include <sstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;
using std::vector;

class DbScan
{
public:
	std::map<int, int> labels;
	vector<Rect>& data;
	int C;
	double eps;
	int mnpts;
	double* dp;
	//memoization table in case of complex dist functions
#define DP(i,j) dp[(data.size()*i)+j]
	DbScan(vector<Rect>& _data, double _eps, int _mnpts) :data(_data)
	{
		C = -1;
		for (int i = 0;i<data.size();i++)
		{
			labels[i] = -99;
		}
		eps = _eps;
		mnpts = _mnpts;
	}
	void run()
	{
		dp = new double[data.size()*data.size()];
		for (int i = 0;i<data.size();i++)
		{
			for (int j = 0;j<data.size();j++)
			{
				if (i == j)
					DP(i, j) = 0;
				else
					DP(i, j) = -1;
			}
		}
		for (int i = 0;i<data.size();i++)
		{
			if (!isVisited(i))
			{
				vector<int> neighbours = regionQuery(i);
				if (neighbours.size()<mnpts)
				{
					labels[i] = -1;//noise
				}
				else
				{
					C++;
					expandCluster(i, neighbours);
				}
			}
		}
		delete[] dp;
	}
	void expandCluster(int p, vector<int> neighbours)
	{
		labels[p] = C;
		for (int i = 0;i<neighbours.size();i++)
		{
			if (!isVisited(neighbours[i]))
			{
				labels[neighbours[i]] = C;
				vector<int> neighbours_p = regionQuery(neighbours[i]);
				if (neighbours_p.size() >= mnpts)
				{
					expandCluster(neighbours[i], neighbours_p);
				}
			}
		}
	}

	bool isVisited(int i)
	{
		return labels[i] != -99;
	}

	vector<int> regionQuery(int p)
	{
		vector<int> res;
		for (int i = 0;i<data.size();i++)
		{
			if (distanceFunc(p, i) <= eps)
			{
				res.push_back(i);
			}
		}
		return res;
	}

	double dist2d(Point2d a, Point2d b)
	{
		return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
	}

	double distanceFunc(int ai, int bi)
	{
		if (DP(ai, bi) != -1)
			return DP(ai, bi);
		Rect a = data[ai];
		Rect b = data[bi];
		/*
		Point2d cena= Point2d(a.x+a.width/2,
		a.y+a.height/2);
		Point2d cenb = Point2d(b.x+b.width/2,
		b.y+b.height/2);
		double dist = sqrt(pow(cena.x-cenb.x,2) + pow(cena.y-cenb.y,2));
		DP(ai,bi)=dist;
		DP(bi,ai)=dist;*/
		Point2d tla = Point2d(a.x, a.y);
		Point2d tra = Point2d(a.x + a.width, a.y);
		Point2d bla = Point2d(a.x, a.y + a.height);
		Point2d bra = Point2d(a.x + a.width, a.y + a.height);

		Point2d tlb = Point2d(b.x, b.y);
		Point2d trb = Point2d(b.x + b.width, b.y);
		Point2d blb = Point2d(b.x, b.y + b.height);
		Point2d brb = Point2d(b.x + b.width, b.y + b.height);

		double minDist = 9999999;

		minDist = min(minDist, dist2d(tla, tlb));
		minDist = min(minDist, dist2d(tla, trb));
		minDist = min(minDist, dist2d(tla, blb));
		minDist = min(minDist, dist2d(tla, brb));

		minDist = min(minDist, dist2d(tra, tlb));
		minDist = min(minDist, dist2d(tra, trb));
		minDist = min(minDist, dist2d(tra, blb));
		minDist = min(minDist, dist2d(tra, brb));

		minDist = min(minDist, dist2d(bla, tlb));
		minDist = min(minDist, dist2d(bla, trb));
		minDist = min(minDist, dist2d(bla, blb));
		minDist = min(minDist, dist2d(bla, brb));

		minDist = min(minDist, dist2d(bra, tlb));
		minDist = min(minDist, dist2d(bra, trb));
		minDist = min(minDist, dist2d(bra, blb));
		minDist = min(minDist, dist2d(bra, brb));
		DP(ai, bi) = minDist;
		DP(bi, ai) = minDist;
		return DP(ai, bi);
	}

	vector<vector<Rect> > getGroups()
	{
		vector<vector<Rect> > ret;
		for (int i = 0;i <= C;i++)
		{
			ret.push_back(vector<Rect>());
			for (int j = 0;j<data.size();j++)
			{
				if (labels[j] == i)
				{
					ret[ret.size() - 1].push_back(data[j]);
				}
			}
		}
		return ret;
	}
};

//	Canny(src_gray, canny_output, thresh, thresh * 2, 3);

//	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
//
//	/// Draw contours
//	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
//	for (int i = 0; i< contours.size(); i++)
//	{
//		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
//	}
//
//	/// Show in a window
//	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
//	imshow("Contours", drawing);

// this class is for OpenCV ParallelLoopBody
//class Parallel_pixel_opencv : public ParallelLoopBody
//{
//private:
//	uchar *p;
//public:
//	Parallel_pixel_opencv(uchar* ptr) : p(ptr) {}
//
//	virtual void operator()(const Range &r) const
//	{
//		for (register int i = r.start; i != r.end; ++i)
//		{
//			//p[i] = (uchar)cos(p[i]);
//			p[i] = 127;
//		}
//	}
//};

void generateRedCyan(Mat &retImage, Mat &image, Mat &depthMap) {
	Mat channel[3];
	//imshow("Original Image", image);

	// The actual splitting.
	split(image, channel);

	//channel[0] = Mat::zeros(image.rows, image.cols, CV_8UC1);//Set blue channel to 0
	//channel[1] = Mat::zeros(image.rows, image.cols, CV_8UC1);//Set green channel to 0
	
	//channel2[2] = Mat::zeros(image.rows, image.cols, CV_8UC1);//Set red channel to 0

	//merge(channel, 3, red);
	//merge(channel2, 3, cyan);

	// Parallel for
	//int width = image.size().width;
	//int height = image.size().height;
	//int nElements = width*height;
	//uchar* p3 = image.data;
	//parallel_for_(Range(0, image.size().area()), Parallel_pixel_opencv(p3));

	// Now I can access each channel separately
	int n, m;
	for (int i = 0; i<image.rows; i++){
		for (int j = 0; j<image.cols; j++){
			n = depthMap.at<uchar>(i,j);
			m = (15 * n)/255;
			if (j < image.cols - m - 1){
				//channel[0].at<uchar>(i, j) = channel[0].at<uchar>(i, j + m);
				image.at<cv::Vec3b>(i, j)[0] = image.at<cv::Vec3b>(i, j + m)[0];
				//channel[1].at<uchar>(i, j) = channel[1].at<uchar>(i, j + m);
				image.at<cv::Vec3b>(i, j)[1] = image.at<cv::Vec3b>(i, j + m)[1];
			}
			if (j > m){
				//channel[2].at<uchar>(i, j) = channel[2].at<uchar>(i, j - m);
				//image.at<cv::Vec3b>(i, j)[2] = image.at<cv::Vec3b>(i, j - m)[2];
			}
			//grouped.at<uchar>(Point(bottomPoint.x, bottomPoint.y))
			//std::cout << three_channels[0].at<uchar>(i, j) << " " << three_channels[1].at<uchar>(i, j) << " " << three_channels[2].at<uchar>(i, j) << std::endl;
		}
	}
	//merge(channel, 3, retImage);

	retImage = image;
}

Point getMaxPoint(std::vector<cv::Point> &points, int vertical, int horizontal) {
	Point max(points.at(0));
	for (auto &point : points) // access by reference to avoid copying
	{
		if ((horizontal * point.x > horizontal * max.x) || (vertical * point.y > vertical * max.y)) {
			max = point;
		}
	}
	return max;
}

Point getBottomPoint(std::vector<cv::Point> &points){
	return getMaxPoint(points, 1, 0);
}

void getDistinctContures(Mat &insertImage, std::vector<std::vector<cv::Point> > &contours, int thresh = 100, int thresh2 = 200, int blurSize = 3) {
	Mat src_gray;
	Mat canny_output;
	std::vector<cv::Vec4i> hierarchy;

	/// Convert image to gray and blur it
	cvtColor(insertImage, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(blurSize, blurSize));

	//imshow("before dilate", src_gray);

	dilate(src_gray, src_gray, vector<Vec6i>());

	//imshow("after dilate",src_gray);

	/// Detect edges using canny
	Canny(src_gray, canny_output, thresh, thresh2, 3);
	
	/// Find contours
	findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0));
}

int main(int argc, char** argv)
{
	/// Initialize vars
	Mat im; 
	RNG rng(12345);
	std::vector<std::vector<cv::Point> > contours;

	/// Load source image and convert it to gray
	if (argc != 2) {
		return -1;
	}
	im = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	/// Check for invalid input
	if (!im.data)                              
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	getDistinctContures(im, contours);

	/// Convert every contour into its bounding box
	vector<Rect> boxes;
	for (size_t i = 0; i < contours.size(); i++)
	{
		Rect r = boundingRect(contours[i]);
		boxes.push_back(r);
	}

	/// Merge bounding boxes that come from same object
	int dbScanDistance = ((im.size().height + im.size().width) /2) * 0.02;//10;//
	DbScan dbscan(boxes, dbScanDistance, 2); 
	dbscan.run();
	
	/// Set linear gradient (255 gray levels)
	Mat lines(im.size(), CV_8U, Scalar(0));
	int col = 0; // goes from 32 to 223
	int rowLen = lines.rows;
	for (int r = 0; r < rowLen; r++)
	{
		col = (191 * r) / rowLen;
		lines.row(r).setTo(col + 32);
	}
	//namedWindow("Linear Gradient", CV_WINDOW_NORMAL);
	//imshow("Linear Gradient", lines);

	Mat grouped = lines;//Mat::zeros(im.size(), CV_8UC3);//

	std::vector<std::vector<cv::Point> > contours2(dbscan.C + 1, std::vector<cv::Point>() );
	vector<Scalar> colors;
	RNG rng2(3);
	int colr = 0;
	for (int i = 0;i <= dbscan.C;i++)
	{
		//colors.push_back(HSVtoRGBcvScalar(rng(255), 255, 255));
		colr = 32 + (191*i) / dbscan.C;
		//colors.push_back(Scalar(256 - colr, 256 - colr, 256 - colr));
	}
	for (int i = 0;i<dbscan.data.size();i++)
	{
		Scalar color;
		if (dbscan.labels[i] == -1)
		{
			color = Scalar(128, 128, 128);
		}
		else
		{
			int label = dbscan.labels[i];
			//color = colors[label];
			contours2[label].insert(contours2[label].end(), contours[i].begin(), contours[i].end());
		}

		//drawContours(grouped, contours, i, color, CV_FILLED);
	}

	// nakon grupisanja kontura iz contours2 probati da svaku konturu posebno nacrtas na slici pa od takve slike izvadis konture i to external?
	Mat slicica = Mat::zeros(grouped.size(), CV_8UC3);
	

	/// Draw merged contours on new image
	Mat drawingContours2 = Mat::zeros(grouped.size(), CV_8UC3);
	for (int i = 0; i< contours2.size(); i++)
	{
		// find bottom pixel on contours2
		Point bottomPoint = getBottomPoint(contours2[i]);
		// get color from pixel beneeth
		colors.push_back(grouped.at<uchar>(Point(bottomPoint.x, bottomPoint.y)));

		Scalar color = Scalar(rng2.uniform(0, 255), rng2.uniform(0, 255), rng2.uniform(0, 255));
		drawContours(drawingContours2, contours2, i, color, CV_FILLED, 8, vector<Vec4i>(), 0, Point());
		//drawContours(drawingContours2, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());

		drawContours(slicica, contours2, i, Scalar(255, 255, 255), CV_FILLED, 8, vector<Vec4i>(), 0, Point());
	}
	//namedWindow("my demo", CV_WINDOW_AUTOSIZE);
	//imshow("my demo", drawingContours2);

	//std::vector<std::vector<cv::Point> > contours3;

	//getDistinctContures(slicica,contours3,100,200,3);
	
	//Mat slicicica = Mat::zeros(grouped.size(), CV_8UC3);
	//for (int i = 0; i< contours3.size(); i++)
	//{
	//	drawContours(slicicica, contours3, i, Scalar(255, 255, 255), CV_FILLED, 8, vector<Vec4i>(), 0, Point());
	//}
	//imshow("slicicica",slicicica);

	//findContours(slicicica, contours3, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	//printf("\nima %d\n", contours3.size());

	//imshow("slicia",slicica);

	/// Fill single object with color
	vector<vector<Point> >hull(contours2.size());
	for (int i = 0; i < contours2.size(); i++)
	{
		convexHull(Mat(contours2[i]), hull[i], false);
	}

	/// Draw contours + hull results
	Mat drawing = Mat::zeros(grouped.size(), CV_8UC3);
	for (int i = 0; i< contours2.size(); i++)
	{
		Scalar color = colors[i];//Scalar(rng2.uniform(0, 255), rng2.uniform(0, 255), rng2.uniform(0, 255));//
		
		drawContours(grouped, hull, i, color, -100, 8, vector<Vec4i>(), 0, Point()); // -- ovo ne izgleda fino zato sto je konveksan hull umjesto konkavan
		//drawContours(grouped, contours2, i, color, -100, 8, vector<Vec4i>(), 0, Point()); // -- ovo ne izgleda fino zato sto radim spajanja slucajno odabranih tacaka
		
		
		// drawContours(drawing, hull, i, color, -100, 8, vector<Vec4i>(), 0, Point());
	}
	/// Show in a window
	//namedWindow("Hull demo", CV_WINDOW_AUTOSIZE);
	//imshow("Hull demo", drawing);

	/// Show starting image
	imshow("im", im);
	/// Show generated depthmap of an image
	imshow("grouped", grouped);
	/// Write depthmap to a file
	imwrite("../data/grouped.jpg", grouped);

	/// Generate red-cyan image
	Mat redCyan = Mat::zeros(grouped.size(), CV_8UC3);
	generateRedCyan(redCyan, im, grouped);

	/// Show red-cyan 3d image
	imshow("redCyan", redCyan);

	waitKey(0);
}
