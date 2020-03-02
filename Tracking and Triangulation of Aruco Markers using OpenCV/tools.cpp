#include "tools.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include "Constants.hpp"

using namespace CVLab;
using namespace cv;
using namespace std;

Mat CVLab::readMatrix(const string &file) {
	// open file to read matrix
	ifstream input(file);
	if (!input.is_open()) {
		throw "file " + file + " could not be opened.";
	}

	// read all lines of the file into the memory
	vector<string> lines;
	for (string line; getline(input, line); ) {
		if (!line.empty()) {
			lines.push_back(line);
		}
	}
	const size_t rows = lines.size();

	// close the file as it is not needed any more
	input.close();

	// count number of columns in the first line
	stringstream lineStream(lines.front());
	size_t cols = 0;
	for (string entry; getline(lineStream, entry, Constants::SeparatorChar); ) {
		if (!entry.empty()) {
			++cols;
		}
	}

	// create matrix and fill with the data from the file
	Mat mat(rows, cols, CV_32F);
	for (size_t row = 0; row < rows; ++row) {
		stringstream line(lines[row]);
		for (size_t col = 0; col < cols; ++col) {
			string entry;
			getline(line, entry, Constants::SeparatorChar);
			mat.at<float>(row, col) = stof(entry);
		}
	}

	// return parsed matrix
	return mat;
}

void CVLab::checkMatrixDimensions(const Mat &mat, int rows, int cols, const string &name) {
	if (((rows >= 0) && (mat.rows != rows)) || ((cols >= 0) && (mat.cols != cols))) {
		throw "invalid matrix dimensions for " + name + " (is " + to_string(mat.rows) + "x" + to_string(mat.cols) + ")";
	}
}

void CVLab::showImage(const Mat &img, const string &title, bool wait) {
	// show the image
	imshow(title, img);

	// and wait for user input
	if (wait) {
		waitKey();
	}
}

void CVLab::showImageMarkers(const Mat &img, const vector<Point2f> &markers, const string &title, bool wait) {
	// convert input image to color image to draw the markers onto it
	Mat markedImage(img.rows, img.cols, CV_8UC3);
	cvtColor(img, markedImage, COLOR_GRAY2BGR);

	// loop over all markers
	for (size_t i = 0; i < markers.size(); ++i) {
		// get color for the current marker
		const Scalar color = Constants::getMarkerColor(i);

		// draw a cross at the position of the marker
		const Point2f &pt = markers[i];
		line(markedImage, Point2f(pt.x - Constants::markerSize, pt.y), Point2f(pt.x + Constants::markerSize, pt.y), color);
		line(markedImage, Point2f(pt.x, pt.y - Constants::markerSize), Point2f(pt.x, pt.y + Constants::markerSize), color);
	}

	// and show the image with the markers
	showImage(markedImage, title, wait);
}

void CVLab::showSequenceMarkers(const vector<Mat> &images, const vector<vector<Point2f> > &markers, const string &title, bool wait) {
	// calculate delay after each frame
	const int delay = 1000 / Constants::frameRate;

	// loop over all frames and show the image with markers
	for (size_t i = 0; i < images.size(); ++i) {
		showImageMarkers(images[i], markers[i], title, false);
		waitKey(delay);
	}

	// wait at the last frame
	if (wait) {
		waitKey();
	}
}

void CVLab::showTriangulation(const vector<vector<Point3f> > &data, const string &title, bool wait) {
	// create white image to show the plot
	Mat plot(640, 640, CV_8UC3);
	plot = Scalar(255, 255, 255);

	// draw axes onto the plot
	const Scalar black(0, 0, 0);
	arrowedLine(plot, Point2i(20, 620), Point2i(620, 620), black, 1, LINE_8, 0, 0.01);
	arrowedLine(plot, Point2i(20, 620), Point2i(20, 20), black, 1, LINE_8, 0, 0.01);
	putText(plot, "y", Point2i(15, 13), FONT_HERSHEY_PLAIN, 1, black);
	putText(plot, "x", Point2i(625, 623), FONT_HERSHEY_PLAIN, 1, black);

	// reorder data structure to hold motion for each marker and determine max and min x and y value
	float xmin = numeric_limits<float>::max();
	float xmax = numeric_limits<float>::min();
	float ymin = numeric_limits<float>::max();
	float ymax = numeric_limits<float>::min();
	vector<vector<Point3f>> markerMotion(data.front().size(), vector<Point3f>(data.size()));
	for (size_t frameIdx = 0; frameIdx < data.size(); ++frameIdx) {
		for (size_t markerIdx = 0; markerIdx < markerMotion.size(); ++markerIdx) {
			markerMotion[markerIdx][frameIdx] = data[frameIdx][markerIdx];

			const auto &marker = markerMotion[markerIdx][frameIdx];
			xmin = min(xmin, marker.x);
			xmax = max(xmax, marker.x);
			ymin = min(ymin, marker.y);
			ymax = max(ymax, marker.y);
		}
	}

	// calculate factor for calculating marker positions in the plot
	const float slopeX = 600 / (xmax - xmin);
	const float slopeY = 600 / (ymax - ymin);

	// plot motion of each marker
	for (size_t markerIdx = 0; markerIdx < markerMotion.size(); ++markerIdx) {
		vector<Point> motion;
		motion.reserve(markerMotion[markerIdx].size());
		for (const auto &pos : markerMotion[markerIdx]) {
			motion.push_back(Point(static_cast<int>((pos.x - xmin) * slopeX + 20.5), static_cast<int>(619.5 - (pos.y - ymin) * slopeY)));
		}

		polylines(plot, motion, false, Constants::getMarkerColor(markerIdx));
	}

	// and show the plot
	showImage(plot, title, wait);
}

void CVLab::writeResult(const string &file, const vector<vector<Point3f> > &result) {
	// open file for writing the result
	ofstream f(file, ios_base::out | ios_base::trunc);
	if (!f.is_open()) {
		throw "could not open file " + file + " for writing result.";
	}

	// write data into the file
	size_t frameIdx = 0;
	for (const auto &frame : result) {
		size_t markerIdx = 0;
		for (const auto &marker : frame) {
			f << to_string(frameIdx) << "," << to_string(markerIdx++) << "," << to_string(marker.x) << "," << to_string(marker.y) << "," << to_string(marker.z) << endl;
		}
		++frameIdx;
	}
}


void CVLab::logMessage(const std::string &message) {
	// get current time
	auto t = time(nullptr);
	auto tm = localtime(&t);

	// print message with current time
#if defined(__GNUC__) && __GNUC__ < 5
	char tstr[9] = "00:00:00";
	strftime(tstr, sizeof(tstr), "%H:%M:%S", tm);
	cout << "[" << tstr << "] " << message << endl;
#else
	cout << "[" << put_time(tm, "%H:%M:%S") << "] " << message << endl;
#endif
}
