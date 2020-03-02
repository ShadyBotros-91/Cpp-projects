#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "Calibration.hpp"

namespace CVLab {
	/**
	 * Class for loading and encapsulating sequence data.
	 * A sequence consists of the images for both cameras and the positions for all markers in the first image in both cameras.
	 * The images can be retrieved from an instance of this class in an array notation obj[camera][image]. The index for camera and image are 0-based.
	 * The marker positions can be retrieved by the method getMarkers which expects the 0-based camera index.
	 */
	class Sequence {
	public:
		/**
		 * Create an object and load the data.
		 *
		 * \param[in] folder The folder to load the sequence data from.
		 * \param[in] c Calibration data.
		 */
		Sequence(const std::string &folder, const Calibration &c);

		/**
		 * Copy Constructor. Creates an object by copying the data from another object. A deep copy of the data is created.
		 *
		 * \param[in] other The object to copy the data from.
		 */
		Sequence(const Sequence &other);

		/**
		 * Get the number of frames in the sequence.
		 */
		size_t getNumberOfFrames() const;

		/**
		 * Get the images for the given camera.
		 *
		 * \param[in] camera Index of the camera to get the images for.
		 */
		const std::vector<cv::Mat> & operator[](unsigned int camera) const;

		/**
		 * Get the marker positions in the first frame.
		 *
		 * \param[in] camera Index of the camera to get the marker positions for.
		 */
		std::vector<cv::Point2f> getMarkers(unsigned int camera) const;

	private:
		/**
		 * Assignment operator. It is disabled as it is not possible to assign constant values.
		 *
		 * \param[in] other The other object that should be assigned to this one.
		 */
		Sequence & operator=(const Sequence &other);

		/**
		 * Read a video from file and save the images in memory.
		 * The images will be converted to grayscale und undistorted.
		 *
		 * \param[in] file The file to read the video from.
		 * \param[out] data The images of the video.
		 * \param[in] K Intrinsics of the camera that recorded the video.
		 * \param[in] distortion Distortion coefficients of the camera that recorded the video.
		 */
		static void readVideo(const std::string &file, std::vector<cv::Mat> &data, const cv::Mat &K, const cv::Mat &distortion);

		/**
		 * Read the marker positions for a camera from a file.
		 *
		 * \param[in] file The file to read the marker positions from.
		 * \param[out] data The marker positions.
		 * \param[in] firstImage The first image of the corresponding video to refine the marker position.
		 */
		static void readMarkers(const std::string &file, std::vector<cv::Point2f> &data, const cv::Mat &firstImage);

		/**
		 * Sort the markers data structure so that the array positions correspond to each other.
		 */
		void sortMarkers();

		/**
		 * Calibration data.
		 */
		const Calibration &calib;

		/**
		 * Undistorted and converted images of the videos.
		 */
		std::vector<cv::Mat> images[2];

		/**
		 * Marker positions of the first frames in the videos.
		 */
		std::vector<cv::Point2f> markers[2];
	};
}