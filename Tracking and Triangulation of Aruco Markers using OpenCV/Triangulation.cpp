#include "Triangulation.hpp"

using namespace CVLab;
using namespace cv;
using namespace std;

Triangulation::Triangulation(const Calibration &c) :
		calib(c) {
}

Triangulation::Triangulation(const Triangulation &other) :
		calib(other.calib) {
}

vector<Point3f> Triangulation::operator()(const vector<Point2f> &markers1,
		const vector<Point2f> &markers2) const {
	Mat fundamentalMatrix = calib.getFundamentalMat(); //fundamental matrix
	Mat k_camera1 = calib.getCamera1(); //Intrinsic matrix for the first camera
	Mat k_camera2 = calib.getCamera2(); //Intrinsic matrix for the second camera
	Mat rotationCam1 = Mat::eye(3, 3, CV_32FC1); //Rotation matrix for the first camera
	Mat translationCam1 = Mat::zeros(3, 1, CV_32FC1); //translation vector for the first camera
	Mat firstCamPos;
	Mat camera1ToWorld = calib.getTransCamera1World(); //Camera 1 coordinates related to the world coordinates

	/*
	 * Adding a fourth row to the transformation matrix of camera 1 to world before dehomogenizing
	 */
	Mat addRowToHomog = (Mat_<float>(1, 4) << 0, 0, 0, 1);
	camera1ToWorld.push_back(addRowToHomog);

	hconcat(rotationCam1, translationCam1, firstCamPos); //Extrinsic matrix for the first camera
	Mat secondCamPos = calib.getTransCamera1Camera2(); //Extrinsic matrix for the second camera
	Mat cam1Projection = k_camera1 * firstCamPos; // Projection matrix for the first camera
	Mat cam2Projection = k_camera2 * secondCamPos; // Projection matrix for the second camera

	//cerr<<camera1ToWorld;

	vector<Point2f> correctMatchesOutput1; //vector for corrected correspondence of markers1
	vector<Point2f> correctMatchesOutput2; //vector for corrected correspondence of markers2

	//The correspondences have to be corrected to fulfill the epipolar constraint
	correctMatches(fundamentalMatrix, markers1, markers2, correctMatchesOutput1,
			correctMatchesOutput2);

	Mat triangulationOutput;

	//After the matches have been corrected, the triangulation can be executed
	triangulatePoints(cam1Projection, cam2Projection, correctMatchesOutput1,
			correctMatchesOutput2, triangulationOutput);

	/*The triangulated points are given in the coordinate system of the first camera, which
	 is not the same as the world coordinate system. Therefore, the points have to be transformed
	 into the world coordinate system*/
	//cerr<<triangulationOutput;
	Mat homogeneousTriangulatedPoints = camera1ToWorld * triangulationOutput;
	vector<Point3f> triangulationResult;

	//Finally, the points have to be dehomogenized and saved into the resbult vector
	convertPointsFromHomogeneous(homogeneousTriangulatedPoints.t(),
			triangulationResult);

	return triangulationResult;
}

vector<vector<Point3f> > Triangulation::operator()(
		const vector<vector<Point2f> > &markers1,
		const vector<vector<Point2f> > &markers2) const {
	// do nothing if there is no data
	if (markers1.empty()) {
		return vector<vector<Point3f>>();
	}

	// check for same number of frames
	if (markers1.size() != markers2.size()) {
		throw "different number of frames";
	}

	// create result vector
	vector<vector<Point3f>> result(markers1.size());

	// triangulate each frame for itself and store result
	for (unsigned int i = 0; i < markers1.size(); ++i) {
		result[i] = (*this)(markers1[i], markers2[i]);
	}

	// and return result
	return result;
}

vector<vector<Point3f> > Triangulation::calculateMotion(
		const vector<vector<Point3f> > &data) {
	//throw "Triangulation::calculateMotion is not implemented";

	/*
	 * As not the absolute positions of the markers are of particular interest, but their motion
	 vectors which are just the marker positions relative to their positions in the first frame.
	 */
	vector<vector<Point3f>> results = data;
	for (size_t i = 0; i < data.size(); i++) {
		for (size_t j = 0; j < NUMBER_OF_MARKERS_PER_FRAME; j++) {
			results[i][j] = data[i][j] - data[0][j];
		}
	}
	return results;
}
