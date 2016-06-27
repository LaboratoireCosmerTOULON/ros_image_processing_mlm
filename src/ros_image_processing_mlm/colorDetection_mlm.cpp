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
		int k_e = 2;	// erode kernel
		int k_d = 7;	// dilate kernel
		// Color segmentation
		int h_min = 0;	// minimum hue
		int h_max = 10;	// maximum hue

		/* Perform contour detection with Canny */
		cv::cvtColor(src, srcGray, cv::COLOR_BGR2GRAY);	// from color to gray					
		cv::blur( srcGray, srcGray, cv::Size(3,3) ); 	// blur image to reduce noise before Canny detector
		cv::Canny( srcGray, cannyOutput, lowThreshold, lowThreshold*ratio, kernel_size );
		cv::findContours(cannyOutput, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		/* Perform color segmentation */
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
	* date   : 06/06/2016
	* 
	* \brief create ROI where rope should be detected
	* \param roi
	* \return roi created. If roi.height = 0 or roi.width = 0, stop seeking for rope
	*/
	void createROI( std::vector<std::vector<int> > linePoints,
					std::vector<float> lineAngles,
					cv::Mat &imgViz,
					int &minLineLength,
					int &maxLineGap,
					std::vector<cv::Rect> &roi )
	{
		/* Initialize variables */
		int dh, dw;		// roi height and width (Attention, the image was rotated by 90 deg)

		std::cout<<"roi_k"<<roi.size()<<std::endl;

		/* Create roi where rope will be detected */
		if (roi.empty())	// if first roi, create a fixed window on image top
		{
			// Set roi height and width
			dw = 0.25*imgViz.cols;
			dh = 30;
			// Set hough line detection parameters for rope detection in image top roi
			minLineLength = floor(0.5*sqrt(2)*dh);
			maxLineGap = floor(0.1*dh);
			roi.push_back( cv::Rect(0.5*imgViz.cols-0.5*dw, 0, dw, dh) ); // create roi (Attention, the image was rotated) 
			cv::rectangle(imgViz, roi.back(), cv::Scalar( 0, 55, 255 ), +1, 4 );	// draw roi
			std::cout<<"roi0draw"<<std::endl;
		}
		else // the others roi are mobile windows with location depending on last rope segment position
		{
			// Set the hough line detection parameters and roi height and width
			dw = 30;
			dh = 30;
			minLineLength = floor(0.1*sqrt(2)*dh);
			maxLineGap = floor(0.75*dh);
			// Control roi overflow
			if(linePoints.back()[3] + 0.75*dh > imgViz.rows)			// overflow by bottom
			{
				std::cout<<"if1"<<std::endl;
				roi.push_back( cv::Rect(0, 0, 0, 0) ); 
			}
			else if(linePoints.back()[3] - 0.25*dh < 0) 			// overflow by top
			{
				std::cout<<"if2"<<std::endl;
				roi.push_back( cv::Rect(0, 0, 0, 0) );
			}
			else if(linePoints.back()[2] + 0.5*dw > imgViz.cols) // overflow by right
			{
				std::cout<<"if3"<<std::endl;
				roi.push_back( cv::Rect(0, 0, 0, 0) );
			}
			else if(linePoints.back()[2] - 0.5*dw < 0) 			// overflow by left
			{
				std::cout<<"if4"<<std::endl;
				roi.push_back( cv::Rect(0, 0, 0, 0) );
			}
			else
			{
				std::cout<<"if5"<<std::endl;
				roi.push_back( cv::Rect(linePoints.back()[2] - 0.5*dw, linePoints.back()[3] - 0.25*dh, dw, dh) );
				std::cout<<"roidraw"<<std::endl;
				cv::rectangle(imgViz, roi.back(), cv::Scalar( 0, 255, 0 ), +1, 4 );	// draw roi
				std::cout<<"roidraw"<<std::endl;
			}
		}
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
	* date   : 27/06/2016
	* 
	* \brief  calculates the angle of the segment of line 'bline' wrt image vertical
	* \param  bline : vector containing start and end point of a segment of line
	* \return the angle alpha between the segment of line 'bline' and the image vertical
	*/
	float lineAngle(std::vector<int> bline)
	{
		int p1x = bline[0];
		int p1y = bline[1];
		int p2x = bline[2];
		int p2y = bline[3];
		float alpha =  atan((float)(p2x - p1x)/(p2y - p1y)); // calculate angle with vertical
		return alpha;
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 03/2016
	* 
	* \brief detects an orange rope by dividing the source image in successive ROIs.
	* \param  an opencv image
	* \return the angle alpha between the image vertical and the rope top part and the line bottom end
	*/
	void lineDetector(cv::Mat imgSrc, cv::Mat &imgMap, cv::Mat &imgViz, float &alpha, float &lineEnd)
	{
		
		/* Prepare source and redering images for processing */
		imgSrc = rotateImg(imgSrc, 90);	// image rotation needed for hough detection
		imgMap  = cv::Mat::zeros( imgSrc.size(), imgSrc.type() );
		imgViz = imgSrc.clone(); 		// visualization post-processing

		/* Perform colored contours detection with Canny */
		cannyDetector(imgSrc, imgMap);	// retrives a binary image with contours of orange spots

		/* Perform line detection */
		// Initialize variables
		std::vector<cv::Rect> roi;					// vector containing ROIs where the rope should be detected 
		std::vector<std::vector<int> > linePoints;	// point coordinates of the segemented line
		std::vector<float> lineAngles;				// anlges of the segemented line
		bool seek = true;							// flag to keep seeking for rope in current ROI
		// find line in ROIs
		std::cout<<std::endl<<"Start line detector"<<std::endl;
		while( seek == true)
		{
			seek = lineROI(imgMap, roi, imgViz, linePoints, lineAngles);
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
		imgMap = rotateImg(imgMap, -90);
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 12/06/2016
	* 
	* \brief  sets line right direction: line begin is near to predecessor line end
	* \param  vector of lines representing segmented rope
	* \return the segment to be added to the vector of lines with right direction
	*/
	void lineDirection(std::vector<float> lineAngles, std::vector<int> &bline)
	{

		/* Invert segment direction
		 * The direction of detected segments from the hough transform is from image left from image rigth
		 * However, when the rope has a negative angle with the image vertical, the direction of the rope segments is from right to lefht,
		 * since the rope is in the image left side. A direction inversion is hence needed 
		*/

		// Extract current segment of line coordinate
		int p1x, p1y, p2x, p2y;
		p1x = bline[0];
		p1y = bline[1];
		p2x = bline[2];
		p2y = bline[3];

		float angle = lineAngles[0];
		// If the current segment is the first one, its direction is necessarily top down
		if(lineAngles.size() < 6 && p2y < p1y)
		{
			// if wrong direction ( p2y < p1y ), invert direction of segment it
			bline.clear();
			bline.push_back(p2x);
			bline.push_back(p2y);
			bline.push_back(p1x);
			bline.push_back(p1y);
		}
		else if( angle < 0.0 )
		{
			// Inversion
			bline[0]= bline[2];
			bline[1]= bline[3];
			bline[2]= p1x;
			bline[3]= p1y;
		}
		std::cout<<"line direction "<<std::endl;
		for(int i = 0; i < lineAngles.size(); i++)
		{
			std::cout<<"angle"<<i<<" "<<lineAngles[i]*180/3.1416<<std::endl;	
		}
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief  given a set of lines detected by the hough transform, calculate best line (average line) points
	* \param  an ROI of an opencv image, vector of hough lines
	* \return coordinates of start and end points of average line and integer (0 echec, 1 success)
	*/
	void bestLine(cv::Mat imgBW, cv::Rect roi, std::vector<cv::Vec4i> lines, std::vector<int> &bline)
	{
		/* Declare variables */
		int p1x, p1y, p2x, p2y; // points of mean line
		int p1x_acc, p1y_acc, p2x_acc, p2y_acc;		// accumulator for calculate mean line
		p1x_acc=0; p1y_acc=0; p2x_acc=0; p2y_acc=0; 
		p1x=0; p1y=0; p2x=0; p2y=0; 				// mean line starting and end point coordinates
		int k = 0;									// counter of useful lines
		int tmp_x, tmp_y;							// temporary variables for ordering line direction
		cv::Point pline_o, pline_f;					// cvPoint for drawing lines

		std::cout<<"Printing all hough lines... "<<std::endl;

		/* Calculate the average position of lines */
		for( size_t i = 0; i < lines.size(); i++ )
		{
			cv::Vec4i l = lines[i]; // extract line from vector of lines 
			std::cout<<"line"<<i<<": "<<l[0]<<" "<<l[1]<<" "<<l[2]<<" "<<l[3]<<std::endl;
			pline_o = cv::Point(l[0] + roi.x, l[1] + roi.y);	// line initial point
			pline_f = cv::Point(l[2] + roi.x, l[3] + roi.y);	// line final point
			cv::line(imgBW, pline_o, pline_f, cv::Scalar(0, 255, 0), 2, 8);			//draw line
			p1x_acc = p1x_acc + pline_o.x;
			p1y_acc = p1y_acc + pline_o.y;
			p2x_acc = p2x_acc + pline_f.x;
			p2y_acc = p2y_acc + pline_f.y;
			k++;
		}
		if (k != 0) // if some line was found useful, compute mean points
		{
			p1x = p1x_acc/k; // line initial point_x
			p1y = p1y_acc/k; // line initial point_y
			p2x = p2x_acc/k; // line end point_x
			p2y = p2y_acc/k; // line end point_y
			bline.push_back(p1x); bline.push_back(p1y); bline.push_back(p2x); bline.push_back(p2y);
		} 
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief  detects lines in a given ROI using hough transformation
	* \param  ROI identifier, source BW image, drawing image
	* \return bool indicating if coordinates of start and end points of average line and its angle wrt the image vertical axis
	*/
	bool lineROI(	cv::Mat imgBW,
					std::vector<cv::Rect> &roi,
					cv::Mat &imgViz,
					std::vector<std::vector<int> > &linePoints,
					std::vector<float> &angles )
	{

		std::cout<<std::endl<<"lineROI"<<std::endl;

		/* Create ROI where rope detection will be performed */
		// Parameters for line detection by Hough
		int minLineLength;	// Minimum line length. Line segments shorter than that are rejected
		int maxLineGap;		// Maximum allowed gap between points on the same line to link them
		bool seek = true; // keep seeking for line only if some line is detected, otherwise stop

		while(seek == true && roi.size() < 50)
		{
			seek = false;
			createROI(linePoints, angles, imgViz, minLineLength, maxLineGap, roi);
			// printing roi...		
			std::cout<<"ROI: "<<roi.back().x<<" "<<roi.back().x+roi.back().width <<" "
							  <<roi.back().y<<" "<<roi.back().y+roi.back().height<<std::endl;

			/* Perform rope detection in roi using Hough algorithm. The rope is modeled by a segment in the roi */
			std::vector<cv::Vec4i> lines;	// vector for line points storage
			if(roi.back().height > 0 ) // if non null roi was created, seek for lines in it
			{
				HoughLinesP(imgBW(roi.back()), lines, 1, CV_PI/180, 20, minLineLength, maxLineGap ); 
				std::cout<<"hough # lines: "<<lines.size()<<std::endl;
			}

			/* Choose most representative line between all lines detected */
			if(!lines.empty()) // if some lines are detected, calculate best line (average line for while...)
			{
				
				/* Calculate most representative line (average line) */
				std::vector<int> bline;
				bestLine(imgViz, roi.back(), lines, bline);

				/* Calculate line angle wrt vertical */
				float alpha = 999; // default value of alpha
				alpha = lineAngle(bline);
				angles.push_back(alpha);	// save angle in vector angles

				/* Correction of line Direction 
				 * Direction from hough : from image left to rigth
				 * Used direction : from image top to bottom 		*/
				std::cout<<"bestLine"<<std::endl;
				lineDirection(angles, bline);
				linePoints.push_back(bline);	// save average line points in vector linePoints
				std::cout<<"bline"<<" "<<bline[0]<<" "<<bline[1]<<" "<<bline[2]<<" "<<bline[3]<<" "<<std::endl;

				/* Plot and print */
				std::cout<<"Angle"<<roi.size()<<": "<< alpha*(180/3.1416) <<std::endl; //print angle
				cv::line(imgViz, cv::Point(bline[0],bline[1]), cv::Point(bline[2],bline[3]), cv::Scalar(0, 0, 255), 2, 8); //draw mean line
				std::cout<<"Printing average line in roi "<<roi.size()-1<<std::endl;
				std::cout<<"Line"<<linePoints.size()<<": "
						 <<linePoints.back()[0]<<" "<<linePoints.back()[1]<<" "<<linePoints.back()[2]<<" "<<linePoints.back()[3]<<" "<<std::endl;
				// keep seeking for rope since one more valid segment was found
				seek = true;
			}
		}
		return seek;
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 03/06/2016
	* 
	* \brief  retrieve the rope pixel coordinates
	* \param  an opencv image source
	* \return processed images for visualizations and pixel coordinates of detected rope
	*/
	void ropePixelCoordinates(cv::Mat imgSrc, cv::Mat &imgMap, cv::Mat &imgViz, std::vector<cv::Point> &locations)
	{
		
		/* Prepare source and redering images for processing */
		imgMap  = cv::Mat::zeros( imgSrc.size(), imgSrc.type() );
		imgViz = imgSrc.clone(); 		// visualization post-processing

		/* Perform colored contours detection with Canny */
		cannyDetector(imgSrc, imgMap);	// retrives a binary image with contours of orange spots

		/* Perform line detection */
		// Initialize variables
		std::vector<cv::Rect> roi_vec;				// vector containing ROIs where the rope should be detected 
		std::vector<std::vector<int> > linePoints;	// point coordinates of the segemented line
		std::vector<float> lineAngles;				// anlges of the segemented line

		// find line in ROIs
		std::cout<<std::endl<<"Start line detector"<<std::endl;
		lineROI(imgMap, roi_vec, imgViz, linePoints, lineAngles);

		/* Create a roi bounding the rope */
		//ropeROI(linePoints, roi_vec);

		/* Retrieve rope pixel coordinates */
		std::vector<cv::Point> loc_roi;		// output, locations of non-zero pixels in current roi
		std::cout<<"roi_vec size "<<roi_vec.size()<<std::endl;
		std::cout<<"linePoints size "<<linePoints.size()<<std::endl;

		std::cout<<"Print rope coordinates"<<std::endl;
		// retieve non-zero pixels inside the rois
		cv::Mat imgRope = cv::Mat::zeros( imgMap.size(), imgMap.type() );
		for(int i = 0; i < roi_vec.size()-1; i++)
		{
			cv::findNonZero(imgMap(roi_vec[i]), loc_roi);
			for(int j = 0; j < loc_roi.size(); j++)
			{
				loc_roi[j].x = loc_roi[j].x + roi_vec[i].x;
				loc_roi[j].y = loc_roi[j].y + roi_vec[i].y;
				locations.push_back(loc_roi[j]);
				imgRope.at<unsigned char>(locations.back().y,locations.back().x) = 255;
			}
		}
		// plot roi_vec in BW image APAGAR DEPOIS
		for(int i = 0; i < roi_vec.size(); i++)
		{
			cv::rectangle(imgMap, roi_vec[i], cv::Scalar(255, 255, 255 ), +1, 4 );	// draw roi
		}
		imgMap = imgRope.clone();
	}


	/*!
	* author : Matheus Laranjeira
	* date   : 10/06/2016
	* 
	* \brief  generates a vector of ROI bounding the rope from its detected points
	* \param  an vectors containg the coordinates of rope segments
	* \return a vector of ROIs bounding the rope
	*/
	void ropeROI(std::vector<std::vector<int> > linePoints, std::vector<cv::Rect> &roi_vec)
	{
		// A COMPLETAR AINDA...
		// clear original roi_vec

		/* Create roi that encloses the detected line points */
		cv::Rect roi;
		int po_x, po_y, pf_x, pf_y;
		for(int i = 0; i < linePoints.size(); i++)
		{
			po_x = linePoints[i][0]; // x-coordinate of segment upper point
			po_y = linePoints[i][1]; // y-coordinate of segment upper point
			pf_x = linePoints[i][2]; // x-coordinate of segment lower point
			pf_y = linePoints[i][3]; // y-coordinate of segment lower point
			
		}
		std::cout<<"linePoints size"<<linePoints.size()<<std::endl;	
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
