#ifndef COLORDETECTION_MLM_HPP
#define COLORDETECTION_MLM_HPP

#include <math.h>
#include <vector>
#include <numeric>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>


namespace ml
{
	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief 	detects orange contours with canny edge detector and image color segmentation in HSV space
	* \param	src		: source image
	* 			imgMap	: BW image with detected orange spots
	* \return 
	*/
	void cannyDetector(cv::Mat src, cv::Mat &imgMap);


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief 	given an image, find orange pixels
	* \param  	imgSrc		: source image
	* 			imgMapMorpho: BW image with orange pixels detected
	* \return 
	*/
	void colorDetector(	const cv::Mat& imgSrc, cv::Mat& imgMapMorpho );


	/*!
	* author : Matheus Laranjeira
	* date   : 06/06/2016
	* 
	* \brief 	create ROIs where rope should be detected
	* \param 	linePoints		: vector containing the line segments coordinates, 
	* 			lineAngles		: vector containing line segments angles wrt vertical,
	* 			imgViz			: image for visualization,
	* 			minLineLengthmin: minimum length of line (line segments shorter than this are rejected)
	* 			maxLineGap		: maximum allowed gap between line segments to treat them as single line
	* 			roi				: vector containing ROIs where rope should be detected
	* \return 
	*/
	void createROI( std::vector<std::vector<int> > linePoints,
					std::vector<float> lineAngles,
					cv::Mat &imgViz,
					int &minLineLength,
					int &maxLineGap,
					std::vector<cv::Rect> &roi );



	/*!
	* author : Matheus Laranjeira
	* date   : 03/2016
	* 
	* \brief	find the non oriented bounding boxe that fits with the larger area contour
	* \param	contours	: contour set of countour points grouped by connexity (see opencv findContour fonction) 
	* \return	bRect		: a non oriented rectangle  
	*/
	cv::Rect findBoundingBoxe(const std::vector<std::vector<cv::Point> > &);


	/*!
	* author : Claire Dune
	* date   : 11/03/2016
	* \brief	find the oriented boxe that fits with the largest area contour
	* \param	contours	: (see cv::findContour)
	* \return 	minRect		: an oriented rectangle
	*/
	cv::RotatedRect findOrientedBoxe(const std::vector<std::vector<cv::Point> > &);


	/*!
	* author : Matheus Laranjeira
	* date   : 03/2016
	* 
	* \brief	finds contours of orange objects in the image
	* \param  	imgSrc		: source image
	* 			imgMapColor	: BW image with orange spots detected
	* 			imgMapMorpho: imgMapColor after some morphological operations
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
	* date   : 27/06/2016
	* 
	* \brief  calculates the angle of the segment of line 'bline' wrt image vertical
	* \param  bline : vector containing start and end point of a segment of line
	* \return the angle alpha between the segment of line 'bline' and the image vertical
	*/
	float lineAngle(std::vector<int> bline);


	/*!
	* author : Matheus Laranjeira
	* date   : 03/2016
	* 
	* \brief	detects an orange rope by dividing the source image in successive ROIs. Calculates 'angle' and 'lineEnd'
	* \param 	imgSrc	: source image
	* 			imgMap	: BW image with detected orange spots
	* 			imgViz	: image for ROI visualization
	* 			alpha	: line angle wrt vertical
	* 			lineEnd	: image coordinates of line end
	* \return
	*/
	void lineDetector(cv::Mat img, cv::Mat &imgMap, cv::Mat &imgViz, float &alpha, float &lineEnd);


	/*!
	* author : Matheus Laranjeira
	* date   : 12/06/2016
	* 
	* \brief  	sets line right direction: line begin is near to predecessor line end
	* \param  	lineAngles	: vector containing the angles of the segments of line wrt vertical
	* 			bline		: the segment of line to be added to the vector of lines with right direction
	* \return 
	*/
	void lineDirection(std::vector<float> lineAngles, std::vector<int> &bline);


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief  given a set of lines detected by the Hough transform, calculate the more representative line (average line)
	* \param  	roi		: vector of ROIs where rope was detected
	* 			lines	: vector of detected lines
	* 			bline	: vector containing the coordinats of more representative line 
	* \return
	*/
	void bestLine(cv::Mat imgBW, cv::Rect roi, std::vector<cv::Vec4i> lines, std::vector<int> &bline);


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief  	detects lines in a given ROI using hough transformation
	* \param  	imgBW		: BW image with line detected points
	* 			roi			: vector with ROIs where line was detected
	* 			imgViz		: image for visualization of ROIs
	* 			linePoints	: image coordinates of detected segments of lines
	* 			angles 		: vector containing angles of detected segments of line wrt vertical
	* \return bool indicating if coordinates of start and end points of average line and its angle wrt the image vertical axis
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
	* \brief 	retrieve the rope pixel coordinates
	* \param  	imgSrc		: source image
	* 			imgMap		: BW with detected rope
	* 			imgViz		: image for visualization of ROIs and detected segments of lines corresponding to the rope
	* 			locations	: rope pixel coordinates
	* 			
	* \return
	*/
	void ropePixelCoordinates(cv::Mat imgSrc, cv::Mat &imgMap, cv::Mat &imgViz, std::vector<cv::Point> &locations);


	/*!
	* author : Matheus Laranjeira
	* date   : 10/06/2016
	* 
	* \brief  generates a vector of ROI bounding the rope from its detected points
	* \param  an vectors containg the coordinates of rope segments
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
