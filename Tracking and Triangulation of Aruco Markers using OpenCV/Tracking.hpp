#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "Calibration.hpp"

namespace CVLab {
	/**
	 * Functor for executing tracking of markers over a sequence of images.
	 */
	class Tracking {
	public:
		/**
		 * Constructor.
		 *
		 * \param[in] c Calibration data.
		 */
		Tracking(const Calibration &c);

		/**
		 * Copy Constructor. Creates an object by copying the data from another object.
		 *
		 * \param[in] other The object to copy the data from.
		 */
		Tracking(const Tracking &other);

		/**
		 * Execute the tracking of the markers on the sequence.
		 *
		 * \param[in] images The images of the sequence to track the markers.
		 * \param[in] initMarkers Positions of the markers in the first frame.
		 * \returns Vector of vector of marker positions. The outer vector has an entry for each frame whereas the inner vector has an entry for each marker in this frame.
		 */
		std::vector<std::vector<cv::Point2f> > operator()(const std::vector<cv::Mat> &images, const std::vector<cv::Point2f> &initMarkers) const;

	private:
		/**
		 * Assignment operator. It is disabled as it is not possible to assign constant values.
		 *
		 * \param[in] other The other object that should be assigned to this one.
		 */
		Tracking & operator=(const Tracking &other);

		/**
		 * Calibration data.
		 */
		const Calibration &calib;
	};
}
