#include "Tracking.hpp"
#include "Sequence.hpp"
using namespace CVLab;
using namespace cv;
using namespace std;

Tracking::Tracking(const Calibration &c) :
		calib(c) {
}

Tracking::Tracking(const Tracking &other) :
		calib(other.calib) {
}

vector<vector<Point2f> > Tracking::operator()(const vector<Mat> &images,
		const vector<Point2f> &initMarkers) const {

	//Creating the markers for the complete set of frames to be tracked
	vector<vector<Point2f>> finalMarkers;
	vector<Point2f> initMarkersCopy;
	//passing the initial markers to every frame
	for (size_t i = 0; i < images.size(); i++) {
		for (size_t j = 0; j < initMarkers.size(); j++) {
			initMarkersCopy.push_back(initMarkers[j]);
		}
		finalMarkers.push_back(initMarkersCopy);
	}
	vector<uchar> status;
	vector<float> err;
	TermCriteria criteria = TermCriteria(
			(TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

	/*Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
	 * Here we iterate till the frame before the last as calcOpticalFlowPyrLk uses the current and the next image in each iteration
	 */
	for (size_t i = 0; i < (images.size() - 1); i++) {
		calcOpticalFlowPyrLK(images[i], images[i + 1], finalMarkers[i],
				finalMarkers[i + 1], status, err, Size(15, 15), 2, criteria);
	}

	return finalMarkers;
}

