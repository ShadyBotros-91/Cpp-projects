#include <opencv2/opencv.hpp>

#include "tools.hpp"
#include "Constants.hpp"
#include "Calibration.hpp"
#include "Sequence.hpp"
#include "Tracking.hpp"
#include "Triangulation.hpp"
#include <string>
#include <iostream>

using namespace CVLab;
using namespace cv;
using namespace std;

int main(int argc, char **argv) {
	try {
		// get calibration folder, sequence folder and output file from command line
		string calibFolder, sequenceFolder, outputFile;
		if (argc == 4) {
			calibFolder = string(argv[1]) + "/";
			sequenceFolder = string(argv[2]) + "/";
			outputFile = string(argv[3]);
		} else {
			cerr
					<< "Please specify folder with calibration data, folder with sequence and output file"
					<< endl;
			return EXIT_FAILURE;
		}

		// load calibration data
		logMessage("load calibration data from " + calibFolder);
		Calibration calib(calibFolder);
		logMessage("loaded calibration data");

		// load sequence
		logMessage("load sequence from " + sequenceFolder);
		Sequence sequence(sequenceFolder, calib);
		logMessage(
				"finished loading sequence with "
						+ to_string(sequence.getNumberOfFrames()) + " frames");

		// track the markers in the sequence
		logMessage("start tracking of markers");

		Tracking trc(calib);
		vector<vector<Point2f>> firstCamMarkers = trc.operator ()(sequence.operator [](0), sequence.getMarkers(0));
		//showSequenceMarkers(sequence.operator [](0),firstCamMarkers, "First sequence of markers",true);
		vector<vector<Point2f>>secondCamMarkers = trc.operator ()(sequence.operator [](1), sequence.getMarkers(1));
		//showSequenceMarkers(sequence.operator [](1),secondCamMarkers,"second sequence of markers",true);

		logMessage("finished tracking of markers");

		// triangulate the marker positions
		logMessage("start triangulation");

		Triangulation trg(calib);
		vector<vector<Point3f>> firstResult = trg.operator ()(firstCamMarkers,secondCamMarkers);
		//showTriangulation(firstResult,"Triangulation result");

		logMessage("finished triangulation");

		// calculate the motion of the markers
		logMessage("calculate motion of markers");

		vector<vector<Point3f>> result = trg.operator ()(firstCamMarkers,secondCamMarkers);
		vector<vector<Point3f>> finalResult = trg.calculateMotion(result);
		//showTriangulation(finalResult,"Motion vector result");

		logMessage("finished calculation of motion of markers");

		// write the result to the output file
		logMessage("write results to " + outputFile);

		writeResult(outputFile,finalResult);

		logMessage("finished writing results");

		// and exit program with code for success
		return EXIT_SUCCESS;
	} catch (const string &err) {
		// print error message and exit program with code for failure
		cerr << err << endl;
		return EXIT_FAILURE;
	}
}
