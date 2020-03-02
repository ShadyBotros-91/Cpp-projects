#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace CVLab {
	/**
	 * Read a matrix from file.
	 *
	 * \param[in] file The file to read the matrix from.
	 */
	cv::Mat readMatrix(const std::string &file);

	/**
	 * Check matrix dimensions and throw exception if they are not correct.
	 * If the value for a dimension is negative, it is not checked.
	 *
	 * \param[in] mat Matrix to check the dimensions for.
	 * \param[in] rows Correct number of rows.
	 * \param[in] cols Correct number of columns.
	 * \param[in] name Name of the matrix for the exception message.
	 */
	void checkMatrixDimensions(const cv::Mat &mat, int rows, int cols, const std::string &name = "given matrix");

	/**
	 * Show an image.
	 *
	 * \param[in] img The image to be shown.
	 * \param[in] title Title of the window. If there exists a window with the same title, the image will be replaced by the new one.
	 * \param[in] wait Flag indicating whether to wait for user input.
	 */
	void showImage(const cv::Mat &img, const std::string &title = "image", bool wait = true);

	/**
	 * Show an image with markers on it.
	 *
	 * \param[in] img The image to be shown.
	 * \param[in] markers The positions of the markers to be shown on the image.
	 * \param[in] title Title of the window. If there exists a window with the same title, the image will be replaced by the new one.
	 * \param[in] wait Flag indicating whether to wait for user input.
	 */
	void showImageMarkers(const cv::Mat &img, const std::vector<cv::Point2f> &markers, const std::string &title = "markers", bool wait = true);

	/**
	 * Show a sequence of images with markers on it.
	 *
	 * \param[in] images The sequence of images.
	 * \param[in] markers Positions of the markers to be shown on the images.
	 * \param[in] title Title of the window. If there exists a window with the same title, the image will be replaced by the new one.
	 * \param[in] wait Flag indicating whether to wait for user input after the last image.
	 */
	void showSequenceMarkers(const std::vector<cv::Mat> &images, const std::vector<std::vector<cv::Point2f> > &markers, const std::string &title = "sequence with markers", bool wait = true);

	/**
	 * Show triangulation result. Only the x- and y-values are shown.
	 * \param[in] data Triangulation result.
	 * \param[in] title Title of the window. If there exists a window with the same title, the image will be replaced by the traingulation result.
	 * \param[in] wait Flag indicating whether to wait for user input.
	 */
	void showTriangulation(const std::vector<std::vector<cv::Point3f> > &data, const std::string &title = "triangulation", bool wait = true);

	/**
	 * Write the triangulation result to a file.
	 *
	 * \param[in] file The file to save the triangulation result to.
	 * \param[in] result Triangulation result to write to the file.
	 */
	void writeResult(const std::string &file, const std::vector<std::vector<cv::Point3f> > &result);

	/**
	 * Print a message to the console prepended with the current time.
	 *
	 * \param[in] message The message to be printed.
	 */
	void logMessage(const std::string &message);
}
