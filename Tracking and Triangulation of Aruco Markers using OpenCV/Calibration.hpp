#pragma once

#include <string>
#include <opencv2/opencv.hpp>

namespace CVLab {
	/**
	 * Class for loading and encapsulating calibration data.
	 * As the calibration data is constant, it is not possible to assign one instance to another.
	 */
	class Calibration {
	public:
		/**
		 * Constructor. Creates an object and loads the data from the given folder.
		 *
		 * \param[in] folder The folder for loading the calibration data.
		 */
		Calibration(const std::string &folder);

		/**
		 * Copy Constructor. Creates an object by copying the data from another object.
		 *
		 * \param[in] other The object to copy the data from.
		 */
		Calibration(const Calibration &other);

		/**
		 * Get the intrinsics of the first camera.
		 */
		cv::Mat getCamera1() const;

		/**
		 * Get the intrinsics of the second camera.
		 */
		cv::Mat getCamera2() const;

		/**
		 * Get the distortion coefficients of the first camera.
		 */
		cv::Mat getDistortion1() const;

		/**
		 * Get the distortion coefficients of the second camera.
		 */
		cv::Mat getDistortion2() const;

		/**
		 * Get the fundamental matrix.
		 */
		cv::Mat getFundamentalMat() const;

		/**
		 * Get the transformation from the first camera to the world coordinate system.
		 */
		cv::Mat getTransCamera1World() const;

		/**
		 * Get the transformation from the first to the second camera.
		 */
		cv::Mat getTransCamera1Camera2() const;

	private:
		/**
		 * Assignment operator. It is disabled as it is not possible to assign constant values.
		 *
		 * \param[in] other The other object that should be assigned to this one.
		 */
		Calibration & operator=(const Calibration &other);

		/**
		 * Intrinsics of the first camera.
		 */
		const cv::Mat camera1;

		/**
		 * Intrinsics of the second camera.
		 */
		const cv::Mat camera2;

		/**
		 * Distortion coefficients of the first camera.
		 */
		const cv::Mat distortion1;

		/**
		 * Distortion coefficients of the second camera.
		 */
		const cv::Mat distortion2;

		/**
		 * Fundamental matrix.
		 */
		const cv::Mat fundamentalMat;

		/**
		 * Transformation from the first camera to the world coordinate system.
		 */
		const cv::Mat transCamera1World;

		/**
		 * Transformation from the first camera to the second camera.
		 */
		const cv::Mat transCamera1Camera2;
	};
}