#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "Calibration.hpp"

#define NUMBER_OF_MARKERS_PER_FRAME 2

namespace CVLab {
	/**
	 * Functor for executing the triangulation. It can be executed on a single frame or a whole sequence.
	 * There is also a method for calculating the motion of the triangulated marker positions. This is simply
	 * done by relating all positions to the position of the first marker in the first frame.
	 */
	class Triangulation {
	public:
		/**
		 * Constructor.
		 *
		 * \param[in] c Calibration data.
		 */
		Triangulation(const Calibration &c);

		/**
		 * Copy Constructor. Creates an object by copying the data from another object. A deep copy of the data is created.
		 *
		 * \param[in] other The object to copy the data from.
		 */
		Triangulation(const Triangulation &other);

		/**
		 * Execute triangulation on a single frame.
		 *
		 * \param[in] markers1 Marker positions in the first camera.
		 * \param[in] markers2 Marker positions in the second camera.
		 * \returns Vector with the triangulated positions of the markers.
		 */
		std::vector<cv::Point3f> operator()(const std::vector<cv::Point2f> &markers1, const std::vector<cv::Point2f> &markers2) const;

		/**
		 * Execute triangulation on a sequence.
		 *
		 * \param[in] markers1 Marker positions for each frame in the first camera.
		 * \param[in] markers2 Marker positions for each frame in the second camera.
		 * \returns Vector with the triangulated marker positions for each frame.
		 */
		std::vector<std::vector<cv::Point3f> > operator()(const std::vector<std::vector<cv::Point2f> > &markers1, const std::vector<std::vector<cv::Point2f> > &markers2) const;

		/**
		 * Calculate the motion of the markers in a sequence.
		 *
		 * \param[in] data The triangulated marker data.
		 * \returns The calculated motion of the markers.
		 */
		static std::vector<std::vector<cv::Point3f> > calculateMotion(const std::vector<std::vector<cv::Point3f> > &data);

	private:
		/**
		 * Assignment operator. It is disabled as it is not possible to assign constant values.
		 *
		 * \param[in] other The other object that should be assigned to this one.
		 */
		Triangulation & operator=(const Triangulation &other);

		/**
		 * Calibration data.
		 */
		const Calibration &calib;
	};
}
