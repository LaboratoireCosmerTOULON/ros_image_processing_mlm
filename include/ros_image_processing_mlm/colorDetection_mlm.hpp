#ifndef COLORDETECTION_MLM_HPP
#define COLORDETECTION_MLM_HPP


#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <geometry_msgs/TwistStamped.h>
#include <math.h>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>


namespace ml
{
	/*!
	* author : Matheus Laranjeira
	* date   : 06/2016
	* 
	* \brief detects orange contours with canny edge detector and image color segmentation in HSV space
	* \param contours, contour set of countour points grouped by connexity (see opencv findContour fonction) 
	* \return bRect, a non oriented rectangle  
	*/
	void cannyDetector(cv::Mat src, cv::Mat &imgMap);


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief given an image, find pixels of specified color 
	* \param  an opencv image
	* \return the corresponding binary image 
	*/
	void colorDetector(	const cv::Mat& imgSrc, cv::Mat& imgMapMorpho );


	/*!
	* author : Matheus Laranjeira
	* date   : 06/06/2016
	* 
	* \brief create ROI where rope should be detected
	* \param roi
	* \return roi created. If roi.height = 0 or roi.width = 0, stop seeking for rope
	*/
	void createROI( std::vector<std::vector<int> > linePoints,
					cv::Mat &imgBW,
					cv::Mat &imgViz,
					int &minLineLength,
					int &maxLineGap,
					std::vector<cv::Rect> &roi );


	/*!
	* author : Matheus Laranjeira
	* date   : 03/2016
	* 
	* \brief find the non oriented bounding boxe that fits with the larger area contour
	* \param contours, contour set of countour points grouped by connexity (see opencv findContour fonction) 
	* \return bRect, a non oriented rectangle  
	*/
	cv::Rect findBoundingBoxe(const std::vector<std::vector<cv::Point> > &);


	/*!
	* author : Claire Dune
	* date   : 11/03/2016
	* \brief find the oriented boxe that fits with the largest area contour
	* \param a contour (see findContour)
	* \return an oriented rectangle
	*/
	cv::RotatedRect findOrientedBoxe(const std::vector<std::vector<cv::Point> > &);


	/*!
	* author : Matheus Laranjeira
	* date   : 03/2016
	* 
	* \brief given
	* \param  an opencv image
	* \return a set of closed contours  
	*/
	std::vector<std::vector<cv::Point> > findColorContour(	const cv::Mat&,
														cv::Mat&,
														cv::Mat&,
														const double&,
														const double&, 
														const double&,
														const double&);


	/*!
	* author : Matheus Laranjeira
	* date   : 03/2016
	* 
	* \brief detects an orange rope in a given image
	* \param  an opencv image
	* \return the angle alpha between the image vertical and the rope top part and the line bottom end
	*/
	void lineDetector(cv::Mat img, cv::Mat &imgMap, cv::Mat &imgViz, float &alpha, float &lineEnd);


	/*!
	* author : Matheus Laranjeira
	* date   : 12/06/2016
	* 
	* \brief  sets line right direction: line begin is near to predecessor line end
	* \param  vector of lines representing segmented rope
	* \return the segment to be added to the vector of lines with right direction
	*/
	void lineDirection(std::vector<std::vector<int> > linePoints, std::vector<float> lineAngles, std::vector<int> &bline);


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief given a set of lines detected by the hough transform, calculate best line (average line) points
	* \param  an ROI of an opencv image, vector of hough lines
	* \return coordinates of start and end points of average line and integer (0 echec, 1 success)
	*/
	void bestLine(cv::Mat imgBW, cv::Rect roi, std::vector<cv::Vec4i> lines, std::vector<int> &bline);


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief  detects lines in a given ROI using hough transformation
	* \param  ROI identifier, source image, drawing image
	* \return coordinates of start and end points of average line and its angle wrt the image vertical
	*/
	bool lineROI(	cv::Mat imgBW,
					std::vector<cv::Rect> &roi,
					cv::Mat &imgViz,
					std::vector<std::vector<int> > &linePoints,
					std::vector<float> &lineAngles );


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief retrieve the rope pixel coordinates
	* \param  an opencv image
	* \return the angle alpha between the image vertical and the rope top part and the line bottom end
	*/
	void ropePixelCoordinates(cv::Mat imgSrc, cv::Mat &imgMap, cv::Mat &imgViz);


	/*!
	* author : Matheus Laranjeira
	* date   : 10/06/2016
	* 
	* \brief  generates a vector of ROI bounding the rope from its detected points
	* \param  an vectors containg some detected rope points
	* \return a vector of ROIs bounding the rope
	*/
	void ropeROI(std::vector<std::vector<int> > linePoints, std::vector<cv::Rect> &roi_vec);


	/*!
	* author : Matheus Laranjeira
	* date   : 06/2016
	* 
	* \brief rotates the given image wrt to its center
	* \param  an opencv image, an rotation angle
	* \return a set of closed contours  
	*/
	cv::Mat rotateImg (cv::Mat src, int angle);
}
#endif
