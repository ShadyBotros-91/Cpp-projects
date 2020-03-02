#include "Calibration.hpp"

#include "tools.hpp"
#include "Constants.hpp"

using namespace CVLab;
using namespace cv;
using namespace std;

Calibration::Calibration(const string &folder) : camera1(readMatrix(folder + Constants::camera1File)),
	                                             camera2(readMatrix(folder + Constants::camera2File)),
												 distortion1(readMatrix(folder + Constants::distortion1File)),
												 distortion2(readMatrix(folder + Constants::distortion2File)),
												 fundamentalMat(readMatrix(folder + Constants::fundamentalMatFile)),
												 transCamera1World(readMatrix(folder + Constants::extCamera1WorldFile)),
												 transCamera1Camera2(readMatrix(folder + Constants::extCamera1Camera2File)) {
	// check validity of matrix dimensions
	checkMatrixDimensions(camera1, 3, 3, "intrinsics of first camera");
	checkMatrixDimensions(camera2, 3, 3, "intrinsics of second camera");
	checkMatrixDimensions(distortion1, 1, 5, "distortion coefficient of first camera");
	checkMatrixDimensions(distortion2, 1, 5, "distortion coefficient of second camera");
	checkMatrixDimensions(fundamentalMat, 3, 3, "fundamental matrix");
	checkMatrixDimensions(transCamera1World, 3, 4, "transformation from first camera to world corrdinate system");
	checkMatrixDimensions(transCamera1Camera2, 3, 4, "transformation from first camera to second camera");
}

Calibration::Calibration(const Calibration &other) : camera1(other.camera1.clone()),
	                                                 camera2(other.camera2.clone()),
													 distortion1(other.distortion1.clone()),
													 distortion2(other.distortion2.clone()),
													 fundamentalMat(other.fundamentalMat.clone()),
													 transCamera1World(other.transCamera1World.clone()),
													 transCamera1Camera2(other.transCamera1Camera2.clone()) {
}

Mat Calibration::getCamera1() const {
	return camera1.clone();
}

Mat Calibration::getCamera2() const {
	return camera2.clone();
}

Mat Calibration::getDistortion1() const {
	return distortion1.clone();
}

Mat Calibration::getDistortion2() const {
	return distortion2.clone();
}

Mat Calibration::getFundamentalMat() const {
	return fundamentalMat.clone();
}

Mat Calibration::getTransCamera1World() const {
	return transCamera1World.clone();
}

Mat Calibration::getTransCamera1Camera2() const {
	return transCamera1Camera2.clone();
}