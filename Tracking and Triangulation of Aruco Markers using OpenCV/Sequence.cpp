#include "Sequence.hpp"
#include "Calibration.hpp"
#include "tools.hpp"
#include "Constants.hpp"

using namespace CVLab;
using namespace cv;
using namespace std;

Sequence::Sequence(const string &folder, const Calibration &c) :
		calib(c) {
	// read both videos
	readVideo(folder + Constants::sequence1File, images[0], calib.getCamera1(),
			calib.getDistortion1());
	readVideo(folder + Constants::sequence2File, images[1], calib.getCamera2(),
			calib.getDistortion2());

	// check if both videos have the same amount of frames
	if (images[0].size() != images[1].size()) {
		throw "both videos have different number of frames";
	}

	// load marker positions for both videos
	readMarkers(folder + Constants::markers1File, markers[0], images[0][0]);
	readMarkers(folder + Constants::markers2File, markers[1], images[1][0]);

	// check if both videos have the same amount of markers
	if (markers[0].size() != markers[1].size()) {
		throw "both videos have different number of markers";
	}

	// sort the markers so that they have the same ordering for both videos
	sortMarkers();
}

Sequence::Sequence(const Sequence &other) :
		calib(other.calib) {
	// loop over all cameras
	for (unsigned int camera = 0; camera < 2; ++camera) {
		// copy images
		images[camera].resize(other.images[camera].size());
		for (unsigned int frame = 0; frame < images[camera].size(); ++frame) {
			images[camera][frame] = other.images[camera][frame].clone();
		}

		// copy marker positions
		markers[camera] = other.markers[camera];
	}
}

size_t Sequence::getNumberOfFrames() const {
	return images[0].size();
}

const vector<Mat> & Sequence::operator[](unsigned int camera) const {
	// check camera index
	if (camera > 1) {
		throw "there are only two cameras";
	}

	// return sequence of images
	return images[camera];
}

vector<Point2f> Sequence::getMarkers(unsigned int camera) const {
	// check camera index
	if (camera > 1) {
		throw "there are only two cameras";
	}

	// return marker positions
	return markers[camera];
}

void Sequence::readVideo(const string &file, vector<Mat> &data, const Mat &K,
		const Mat &distortion) {
	// open video file
	VideoCapture vid(file);
	if (!vid.isOpened()) {
		throw "could not open video file " + file;
	}

	// get number of frames from the video file
	const unsigned int numberOfFrames = static_cast<unsigned int>(vid.get(
			CAP_PROP_FRAME_COUNT));

	// resize vector to number of frames
	data.clear();
	data.resize(numberOfFrames);

	// load images from video
	for (unsigned int i = 0; i < numberOfFrames; ++i) {
		Mat img, gray, undistorted;

		// load next frame
		vid >> img;

		// convert frame to grayscale
		cvtColor(img, gray, COLOR_BGR2GRAY);

		// undistort the image
		undistort(gray, undistorted, K, distortion);

		// save the undistorted image in the vector
		undistorted.copyTo(data[i]);
	}
}

void Sequence::readMarkers(const string &file, vector<Point2f> &data,
		const Mat &firstImage) {
	// read raw data from file
	Mat markerData = readMatrix(file);

	// check matrix dimension for validity
	checkMatrixDimensions(markerData, -1, 2, "marker positions");

	// resize vector to take marker positions
	data.clear();
	data.resize(markerData.rows);

	// save marker positions in the vector
	for (int i = 0; i < markerData.rows; ++i) {
		data[i].x = markerData.at<float>(i, 0);
		data[i].y = markerData.at<float>(i, 1);
	}

	// and refine the marker positions
	cornerSubPix(firstImage, data, Constants::markerRefinementWindowSize,
			Constants::markerRefinementZeroZone,
			Constants::markerRefinementCriteria);
}

void Sequence::sortMarkers() {


	//Reading the fundamental matrix
	Mat fund = calib.getFundamentalMat();

	//checking the first point in the first image for correspondence
		Mat x1 = (Mat_<float>(3, 1) << markers[0][0].x, markers[0][0].y, 1, CV_32FC1);

	//Checking the first point in the second image
	Mat x1_prime_t =(Mat_<float>(1, 3) << markers[1][0].x, markers[1][0].y, 1, CV_32FC1);

	//Satisfying the correspondence condition for comparing two image points
	Mat result = x1_prime_t * fund * x1;

	//A scalar to compare the result of the correspondence condition
	Mat comparingResult = (Mat_<float>(1, 1) << 0);

	//A matrix for saving the result of the comparison
	Mat checkingResult;

	absdiff(result, comparingResult, checkingResult);

	//Checking if the difference of results is greater than a threshold of 5 , then the order of markers should be changed
	if (checkingResult.at<float>(0, 0) > 5) {
		Point2f temp;
		temp=markers[1][0];
		markers[1][0]=markers[1][1];
		markers[1][1]=temp;
	}
	//Checking that the markers' orders have been changed
	//showImageMarkers(images[0][1], markers[0]);
	//showImageMarkers(images[1][1], markers[1]);
}
