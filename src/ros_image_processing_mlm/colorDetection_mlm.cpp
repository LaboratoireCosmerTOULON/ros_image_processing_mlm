#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <geometry_msgs/TwistStamped.h>
#include <math.h>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <ros_image_processing_mlm/colorDetection_mlm.hpp>

namespace ml
{
	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief  detects orange contours with canny edge detector and image color segmentation in HSV space
	* \param  source image
	* \return a binary image imgMap with detected orange spots
	*/
	void cannyDetector(cv::Mat src, cv::Mat &imgMap)
	{
		/* Initialize variables */
		// Matrix and vectors for image processing and contours detection
		cv::Mat srcGray, srcHsv, cannyOutput;
		std::vector<std::vector<cv::Point> > contours;
		// Canny parameters
		int ratio = 3;
		int kernel_size = 3;
		int lowThreshold = 0;
		// Morphological operations
		int k_e = 1;	// erode kernel
		int k_d = 3;	// dilate kernel
		// Color segmentation
		int h_min = 0;	// minimum hue
		int h_max = 10;	// maximum hue

		/* Perform contour detection with Canny **/
		cv::cvtColor(src, srcGray, cv::COLOR_BGR2GRAY);	// from color to gray					
		cv::blur( srcGray, srcGray, cv::Size(3,3) ); 	// blur image to reduce noise before Canny detector
		cv::Canny( srcGray, cannyOutput, lowThreshold, lowThreshold*ratio, kernel_size );
		cv::findContours(cannyOutput, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		/** Perform color segmentation **/
		cv::Rect bRect = cv::Rect(0,0,0,0);
		cv::cvtColor(src, srcHsv, cv::COLOR_BGR2HSV);		// from color to hsv for color detection
		cv::cvtColor(imgMap, imgMap, cv::COLOR_BGR2GRAY);	// transform to binary image (needed for inRange)					
		for( int i = 0; i< contours.size(); i++ )
		{
			if (cv::contourArea(contours[i]) > 0.00001*src.rows*src.cols)
			{
				bRect = cv::boundingRect(contours[i]);
				cv::inRange(srcHsv(bRect), cv::Scalar(h_min,50,50), cv::Scalar(h_max,255,255), imgMap(bRect));
			}
		}

		/** Perform morphological operations to reduce noise **/
		cv::erode(imgMap, imgMap, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k_e,k_e)));
		cv::dilate(imgMap, imgMap, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k_d, k_d)));
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief given an image, find pixels of specified color 
	* \param  an opencv image
	* \return the corresponding binary image 
	*/
	void colorDetector(	const cv::Mat& imgSrc, cv::Mat& imgMapMorpho )
	{
		/* Initialize variables */
		// Matrix for image processing and contours detection
		cv::Mat imgHsv, imgMapColor;
		imgMapColor  = cv::Mat::zeros( imgSrc.size(), imgSrc.type() );
		imgHsv       = cv::Mat::zeros( imgSrc.size(), imgSrc.type() );
		// Morphological operations
		const int m_e = 1;	// erode kernel
		const int m_d = 3;	// dilate kernel
		// Color segmentation
		const int h_min = 0;	// minimum hue
		const int h_max = 10;	// maximum hue

		/* Color Detection */
		// convertion from color to hsv
		cv::cvtColor(imgSrc, imgHsv, cv::COLOR_BGR2HSV);
		// Select the desired color and build the binary image map
		cv::inRange(imgHsv, cv::Scalar(h_min,0,0), cv::Scalar(h_max,255,255), imgMapColor);
		
		/* Perform morphological operations */
		// dilate the detected area and build the map
		cv::erode(imgMapColor , imgMapMorpho , getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(m_e,m_e)));
		cv::dilate(imgMapMorpho, imgMapMorpho, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(m_d,m_d)));	
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 03/2016
	* 
	* \brief find the non oriented bounding boxe that fits with the larger area contour
	* \param contours, contour set of countour points grouped by connexity (see opencv findContour fonction) 
	* \return bRect, a non oriented rectangle  
	*/
	cv::Rect findBoundingBoxe(const std::vector<std::vector<cv::Point> > &contours)
	{
		cv::Rect bRect(0,0,0,0); 
		double largest_area = 0.0; 
		for( int i = 0; i< contours.size(); i++ )
		{
			double a = contourArea( contours[i],false);  //  Find the area of contour
			if(a>largest_area)
			{
				largest_area = a;
				bRect        = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
			}
		}
		return bRect;
	}


	/*!
	* author : Claire Dune
	* date   : 11/03/2016
	* \brief find the oriented boxe that fits with the largest area contour
	* \param a contour (see findContour)
	* \return an oriented rectangle
	*/
	cv::RotatedRect findOrientedBoxe(const std::vector<std::vector<cv::Point> > &contours)
	{
		/// Find the rotated rectangles the biggest contour
		cv::RotatedRect minRect;
		double largest_area=0.0; 
		for( int i = 0; i < contours.size(); i++ )
		{ 
			double area = cv::contourArea(contours[i],false);
			if( area > largest_area )
			{
				largest_area = area;
				minRect      = cv::minAreaRect( cv::Mat(contours[i]) );
			}
		}
		return minRect;
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 03/2016
	* 
	* \brief given
	* \param  an opencv image
	* \return a set of closed contours  
	*/
	std::vector<std::vector<cv::Point> > findColorContour(	const cv::Mat& imgSrc, cv::Mat& imgMapColor, cv::Mat& imgMapMorpho)
	{
		/* Initialize variables */
		// Matrix for image processing and contours detection
		cv::Mat imgHsv;
		imgMapColor  = cv::Mat::zeros( imgSrc.size(), imgSrc.type() );
		imgHsv       = cv::Mat::zeros( imgSrc.size(), imgSrc.type() );
		// Morphological operations
		const int m_e = 15;	// erode kernel
		const int m_d = 25;	// dilate kernel
		// Color segmentation
		const int h_min = 0;	// minimum hue
		const int h_max = 10;	// maximum hue

		/* Color Detection */
		// convertion from color to hsv
		cv::cvtColor(imgSrc, imgHsv, cv::COLOR_BGR2HSV);
		// Select the desired color and build the binary image map
		cv::inRange(imgHsv, cv::Scalar(h_min,50,50), cv::Scalar(h_max,255,255), imgMapColor);
		
		/* Perform morphological operations */
		// dilate the detected area and build the map
		cv::erode(imgMapColor , imgMapMorpho , getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(m_e,m_e)));
		cv::dilate(imgMapMorpho, imgMapMorpho, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(m_d,m_d)));

		/* Contours detection */
		// Find contours of the white area in the image map
		std::vector<std::vector<cv::Point> > contours;
		cv::findContours(imgMapMorpho.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		return contours;	
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 03/2016
	* 
	* \brief detects an orange rope by dividing the source image in successive ROIs.
	* \param  an opencv image
	* \return the angle alpha between the image vertical and the rope top part and the line bottom end
	*/
	void lineDetector(cv::Mat imgSrc, cv::Mat &imgViz, float &alpha, float &lineEnd)
	{
		
		/* Prepare source and redering images for processing */
		imgSrc = rotateImg(imgSrc, 90);	// image rotation needed for hough detection
		cv::Mat imgMap  = cv::Mat::zeros( imgSrc.size(), imgSrc.type() );
		imgViz = imgSrc.clone(); 		// visualization post-processing

		/* Perform colored contours detection with Canny */
		cannyDetector(imgSrc, imgMap);	// retrives a binary image with contours of orange spots

		/* Perform line detection */
		int roi_k; // roi counter (from 0 to 3)
		std::vector<std::vector<int> > linePoints;	// point coordinates of the segemented line
		std::vector<float> lineAngles;				// anlges of the segemented line
	
		// find line in ROIs
		roi_k = 0; 	std::cout<<std::endl<<"Start line detector"<<std::endl;
		while(roi_k < 4)
		{
			lineROI(roi_k, imgMap, imgViz, linePoints, lineAngles);
			roi_k++;
		}
		if(linePoints.size() > 0)	// if some line was found, save it
		{
			std::cout<<"Printing lines angles ending points... "<<std::endl;
			for(int i = 0; i < linePoints.size(); i++){
				std::cout<<"Line "<<i<<": "<<linePoints[i][2]<<" "<<linePoints[i][3]<<" "<<std::endl;
				std::cout<<"Angle "<<i<<": "<<lineAngles[i]<<std::endl;
			}
			alpha = lineAngles[0];	// save angle of first line segment wrt image vertical axis
			lineEnd = linePoints[linePoints.size() - 1][2];
		}
		else
		{
		std::cout<<"No line found"<<std::endl;
			alpha = 999;
			lineEnd = 999;
		}

		// rotate image back to original position for right visualization
		imgViz = rotateImg(imgViz, -90);
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief given a set of lines detected by the hough transform, calculate best line (average line) points
	* \param  an ROI of an opencv image, vector of hough lines
	* \return coordinates of start and end points of average line and integer (0 echec, 1 success)
	*/
	int lineParam(cv::Mat imgBW, cv::Rect roi, std::vector<cv::Vec4i> lines, int &p1x, int &p1y, int &p2x, int &p2y)
	{
		/* Declare variables */
		int p1x_acc, p1y_acc, p2x_acc, p2y_acc;
		p1x_acc=0; p1y_acc=0; p2x_acc=0; p2y_acc=0; // accumulator for calculate mean line
		p1x=0; p1y=0; p2x=0; p2y=0; 				// mean line starting and end point coordinates
		int k = 0;									// counter of useful lines
		cv::Point pline_o, pline_f;					// cvPoint for drawing lines

		std::cout<<"ROI: "<<roi.x<<" "<<roi.x+roi.width<<" "<<" "<<roi.y<<" "<<roi.y+roi.height<<" "<<std::endl;
		std::cout<<"Printing all hough lines... "<<std::endl;

		/* Calculate average position of lines */
		for( size_t i = 0; i < lines.size(); i++ )
		{
			cv::Vec4i l = lines[i];
			std::cout<<"line"<<i<<": "<<l[0]<<" "<<l[1]<<" "<<l[2]<<" "<<l[3]<<std::endl;
			if (l[0] < l[2] && l[0] < 30) // calculate mean only for vertical lines that start on image top
			{
				pline_o = cv::Point (l[0] + roi.x, l[1] + roi.y);	// line initial point
				pline_f = cv::Point (l[2] + roi.x, l[3] + roi.y);	// line final point
				cv::line(imgBW, pline_o, pline_f, cv::Scalar(0, 255, 0), 2, 8);			//draw line
				p1x_acc = p1x_acc + pline_o.x;
				p1y_acc = p1y_acc + pline_o.y;
				p2x_acc = p2x_acc + pline_f.x;
				p2y_acc = p2y_acc + pline_f.y;
				k++;
			}
		}
		if (k != 0) // if some line was found useful, compute mean points
		{
			p1x = p1x_acc/k; // line initial point_x
			p1y = p1y_acc/k; // line initial point_y
			p2x = p2x_acc/k; // line end point_x
			p2y = p2y_acc/k; // line end point_y
		} 
		return k;
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief  detects lines in a given ROI using hough transformation
	* \param  ROI identifier, source BW image, drawing image
	* \return coordinates of start and end points of average line and its angle wrt the image vertical axis
	*/
	void lineROI(int &roi_k, cv::Mat &imgBW, cv::Mat &imgViz, std::vector<std::vector<int> > &linePoints, std::vector<float> &lineAngles)
	{
		cv::Rect roi;		//roi where will be performed the line detection
		int dh, dw;			// roi size
		dw = 120; dh = 120;	// standard roi size

		std::cout<<"roi_k"<<roi_k<<std::endl;

		/* Create roi where rope will be detected */
		if (roi_k == 0)	// if first roi, create a fixed window on image top
		{
			dw	= 120; dh = 320;											// standard size of first roi
			roi	= cv::Rect(0, 0.5*imgBW.rows-0.5*dh, dw, dh); 
			cv::rectangle(imgViz, roi, cv::Scalar( 0, 55, 255 ), +1, 4 );	// draw roi
			std::cout<<"roi0draw"<<std::endl;
		}
		else // the others roi are mobile windows with location depending on last rope segment position
		{

			// control roi overflow for others roi
			if(linePoints[roi_k-1][3] + 0.5*dh > imgBW.rows)
			{
				std::cout<<"if1"<<std::endl;
				roi = cv::Rect(linePoints[roi_k-1][2], linePoints[roi_k-1][3], dw, imgBW.rows - linePoints[roi_k-1][3]);
			}
			else if(linePoints[roi_k-1][3] - 0.5*dh < 0)
			{
				std::cout<<"if2"<<std::endl;
				roi = cv::Rect(linePoints[roi_k-1][2], 0, dw, dh);
			}
			else
			{
				std::cout<<"if3"<<std::endl;
				roi = cv::Rect(linePoints[roi_k-1][2], linePoints[roi_k-1][3] - 0.5*dh, dw, dh);
			}
			std::cout<<"roidraw"<<std::endl;
			cv::rectangle(imgViz, roi, cv::Scalar( 0, 255, 0 ), +1, 4 );	// draw roi
			std::cout<<"roidraw"<<std::endl;
		}

		/* Perform rope detection in roi through Hough algorithm. The rope is modeled by a segment in the roi */
		std::vector<cv::Vec4i> lines;	// vector for line points storage
		std::cout<<"hough"<<std::endl;
		HoughLinesP(imgBW(roi), lines, 1, CV_PI/180, 50, 50, 20 ); 
		std::cout<<"hough"<<std::endl;
		std::cout<<"hough # lines: "<<lines.size()<<std::endl;

		if (lines.size() > 0) // if some lines are detected, calculate best line (average line for while...)
		{
			int p1x, p1y, p2x, p2y; // points of mean line
			if (lineParam(imgViz, roi, lines, p1x, p1y, p2x, p2y) > 0) // if a useful line was found...
			{
				float alpha = 999; // default value of alpha
				std::vector<int> linerow;
				linerow.push_back(p1x); linerow.push_back(p1y); linerow.push_back(p2x); linerow.push_back(p2y);
				std::cout<<"linerow"<<" "<<linerow[0]<<" "<<linerow[1]<<" "<<linerow[2]<<" "<<linerow[3]<<" "<<std::endl;
				linePoints.push_back(linerow);	// save average line points in vector linePoints
				alpha =  atan((float)(p2y-p1y)/(p2x-p1x)); // calculate angle with vertical
				lineAngles.push_back(alpha);	// save angle in vector lineAngles
				std::cout<<"Angle"<<roi_k<<": "<< alpha*(180/3.1416) <<std::endl; //print angle
				cv::line(imgViz, cv::Point (p1x,p1y), cv::Point(p2x,p2y), cv::Scalar(0, 0, 255), 2, 8); //draw mean line
				std::cout<<"Printing average line in roi "<<roi_k<<std::endl;
				for(int i = roi_k; i < roi_k+1; i++)
				{
						std::cout<<"Line"<<i<<": "<<linePoints[i][0]<<" "<<linePoints[i][1]<<" "<<linePoints[i][2]<<" "<<linePoints[i][3]<<" "<<std::endl;
				}
			}
			else{roi_k = 4;}// if no useful line was found, stop seeking for line...
		}
		else{roi_k = 4;}// if no line was found by Hough, stop seeking for line...
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 06/2016
	* 
	* \brief rotates the given image wrt to its center
	* \param  an opencv image, an rotation angle
	* \return a set of closed contours  
	*/
	cv::Mat rotateImg (cv::Mat src, int angle)
	{
		cv::Point2f center(src.cols/2.0F, src.rows/2.0F);
		cv::Mat rot = getRotationMatrix2D(center, angle, 1.0);
		cv::Mat dst;
		cv::Rect bbox = cv::RotatedRect(center,src.size(), angle).boundingRect();
		// adjust transformation matrix
		rot.at<double>(0,2) += bbox.width/2.0 - center.x;
		rot.at<double>(1,2) += bbox.height/2.0 - center.y;
		cv::warpAffine(src, dst, rot, bbox.size());
		return dst;
	}
}
