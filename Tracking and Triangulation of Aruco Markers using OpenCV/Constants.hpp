#pragma once
// CV_TERMCRIT_EPS replace by cv::TermCriteria::EPS
// CV_TERMCRIT_ITER replace by cv::TermCriteria::MAX_ITER
// CV_GRAY2BGR replace by COLOR_GRAY2BGR

#include <string>
#include <opencv2/opencv.hpp>

namespace CVLab {
	namespace Constants {
		/**
		 * The character that is used in the files to separate matrix columns.
		 */
		const char SeparatorChar = ',';

		/**
		 * File name of the intrinsics of the first camera.
		 */
		const std::string camera1File("K1.csv");

		/**
		 * File name of the intrinsics of the second camera.
		 */
		const std::string camera2File("K2.csv");

		/**
		 * File name of the distortion coefficients of the first camera.
		 */
		const std::string distortion1File("D1.csv");

		/**
		 * File name of the distortion coefficients of the second camera.
		 */
		const std::string distortion2File("D2.csv");

		/**
		 * File name of the fundamental matrix.
		 */
		const std::string fundamentalMatFile("F.csv");

		/**
		 * File name of the transformation from the first camera to the world coordinate system.
		 */
		const std::string extCamera1WorldFile("T_C1_W.csv");

		/**
		 * File name of the transformation from the first camera to the second camera.
		 */
		const std::string extCamera1Camera2File("T_C1_C2.csv");

		/**
		 * File name of the video file from the first camera.
		 */
		const std::string sequence1File("MarkerSequence_1.avi");

		/**
		 * File name of the video file from the second camera.
		 */
		const std::string sequence2File("MarkerSequence_2.avi");

		/**
		 * File name of the initial marker positions for the first camera.
		 */
		const std::string markers1File("markers1.csv");

		/**
		 * File name of the initial marker positions for the second camera.
		 */
		const std::string markers2File("markers2.csv");

		/**
		 * Size of the search window in refinement of the marker positions.
		 */
		const cv::Size markerRefinementWindowSize(5, 5);

		/**
		 * Size of the zero zone in refinement of the marker positions.
		 */
		const cv::Size markerRefinementZeroZone(-1, -1);

		/**
		 * Termination criteria for refinement of markers positions.
		 */
		const cv::TermCriteria markerRefinementCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 40, 0.001);

		/**
		 * Size of the cross when drawing markers on an image.
		 */
		const unsigned int markerSize = 5;

		/**
		 * Frame rate in fps when showing a sequence with markers on it.
		 */
		const unsigned short frameRate = 4;

		/**
		 * Get the color for the marker with the given index.
		 *
		 * \param[in] index Index of the marker.
		 */
		template<typename T> cv::Scalar getMarkerColor(T index) {
			T m = index % 3;
			switch (m) {
			case 0:
				return cv::Scalar(0, 0, 255);
			case 1:
				return cv::Scalar(0, 255, 0);
			default:
				return cv::Scalar(255, 0, 0);
			}
		}
	}
}