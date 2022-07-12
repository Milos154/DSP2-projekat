
#include "Enums.hpp"

#include <visualizer.hpp>

#include <iostream>
#include <cstdlib>
#include <vector>
using namespace std;

#include <unistd.h>

#define DEBUG(x) do{ std::cout << #x << " = " << x << std::endl; }while(0)

#include "opencv2/opencv.hpp"
using namespace cv;

int main() {
	//Mat src = cv::imread("data/stop_sign.jpg");
	Mat src = cv::imread("data/one_way.png");
	//Mat src = cv::imread("data/black_spot_sign.jpg");
	//Mat src = cv::imread("data/crosswalk_sign.jpg");
	//Mat src = cv::imread("data/danger_sign.jpg");
	//Mat src = cv::imread("data/highway_exit_sign.jpg");
	//Mat src = cv::imread("data/no_parking.png");
	//Mat src = cv::imread("data/priority_sign.jpg");
	//Mat src = cv::imread("data/pedestrian_zone_sign.webp");
	//Mat src = cv::imread("data/proceed_straight_sign.webp");

	if(src.empty()){
		throw runtime_error("Cannot open image!");
	}

	visualizer::load_cfg("data/main.visualizer.yaml");

	const uint16_t width = 10;
	const uint16_t height = 20;
	static uint8_t pix[width*height*3];
	for(uint16_t y = 0; y < height; y++){
		for(uint16_t x = 0; x < width; x++){
			uint32_t i = (y*width + x)*3;
			// Red.
			pix[i+0] = 0;
			pix[i+1] = 0;
			pix[i+2] = 255;
		}
	}

	for(uint16_t y = 3; y < height-3; y++){
		for(uint16_t x = 3; x < width-3; x++){
			uint32_t i = (y*width + x)*3;
			// Blue.
			pix[i+0] = 255;
			pix[i+1] = 0;
			pix[i+2] = 0;
		}
	}




	int th_start_h0 = 0, th_start_s = 0, th_start_v = 0;
	int th_stop_h0 = 180, th_stop_s = 255, th_stop_v = 255;
  int th_start_h1 = 0, th_stop_h1 = 180;

  
  visualizer::slider::slider(
    "/win0/upper_half/upper_rigth_corner/th_start_h0",
    th_start_h0,
    [&](int& value){
    }
  );

  visualizer::slider::slider(
    "/win0/upper_half/upper_rigth_corner/th_start_h1",
    th_start_h1,
    [&](int& value){
    }
  );

  visualizer::slider::slider(
    "/win0/upper_half/upper_rigth_corner/th_start_s",
    th_start_s,
    [&](int& value){
    }
  );

  visualizer::slider::slider(
    "/win0/upper_half/upper_rigth_corner/th_start_v",
    th_start_v,
    [&](int& value){
    }
  );

  visualizer::slider::slider(
    "/win0/upper_half/upper_rigth_corner/th_stop_h0",
    th_stop_h0,
    [&](int& value){
    }
  );

  visualizer::slider::slider(
    "/win0/upper_half/upper_rigth_corner/th_stop_h1",
    th_stop_h1,
    [&](int& value){
    }
  );

  visualizer::slider::slider(
    "/win0/upper_half/upper_rigth_corner/th_stop_s",
    th_stop_s,
    [&](int& value){
    }
  );

  visualizer::slider::slider(
    "/win0/upper_half/upper_rigth_corner/th_stop_v",
    th_stop_v,
    [&](int& value){
    }
  );

  
  Mat src_hsv;
  cv::cvtColor(src, src_hsv, cv::COLOR_BGR2HSV);
 
  Mat src_hsv_channels[3];
  cv::split(src_hsv, src_hsv_channels);

  Mat src_hsv_channels_rgb[3];
  cv::cvtColor(src_hsv_channels[0], src_hsv_channels_rgb[0], cv::COLOR_GRAY2BGR);
  cv::cvtColor(src_hsv_channels[1], src_hsv_channels_rgb[1], cv::COLOR_GRAY2BGR);
  cv::cvtColor(src_hsv_channels[2], src_hsv_channels_rgb[2], cv::COLOR_GRAY2BGR);
  
	visualizer::img::show(
		"h",
		src_hsv_channels_rgb[0]
	);

	visualizer::img::show(
		"s",
		src_hsv_channels_rgb[1]
	);

	visualizer::img::show(
		"v",
		src_hsv_channels_rgb[2]
	);

  Mat selected_low, selected_high;
  Mat selected, selected_rgb;

  visualizer::img::show(
    "src", 
    src
  );

	while(true){
    cv::inRange(src_hsv, 
        Scalar(th_start_h0, th_start_s, th_start_v),
        Scalar(th_stop_h0, th_stop_s, th_stop_v), selected_low);

    cv::inRange(src_hsv, 
        Scalar(th_start_h1, th_start_s, th_start_v),
        Scalar(th_stop_h1, th_stop_s, th_stop_v), selected_high);

    cv::bitwise_or(selected_low, selected_high, selected);
    cv::cvtColor(selected, selected_rgb, cv::COLOR_GRAY2BGR);

    visualizer::img::show(
      "in_range", 
      selected_rgb
    );

    
    Mat selected_morph;
    Mat kernel = Mat::ones(5, 5, CV_8U);
    cv::morphologyEx(selected, selected_morph, cv::MORPH_CLOSE, kernel);
    
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(selected_morph, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

   
    if(contours.size() == 0) {
      visualizer::slider::update();
      sleep(1);
      continue;
    }

    double biggest_area = 0;
    int biggest_idx = 0;
    for(int i=0; i<contours.size(); ++i) {
      if(biggest_area < cv::contourArea(contours[i])) {
        biggest_idx = i;
        biggest_area = cv::contourArea(contours[i]);
      }
    }

    
    std::vector<cv::Point> hull;
    cv::convexHull(contours[biggest_idx], hull, true);

    
    Mat with_contour = src.clone();
    polylines(with_contour, hull, true, Scalar(0, 255, 0), 3, LINE_AA);

    visualizer::img::show(
      "with_contour", 
      with_contour
    );
    
    Mat mask = Mat::zeros(src.rows, src.cols, CV_8U);
    cv::fillConvexPoly(mask, hull, Scalar(255));

    Mat bg_removed;
    src.copyTo(bg_removed, mask);

    
    Rect bb = cv::boundingRect(hull);

   
    Mat outImg = bg_removed(
        Range(bb.y, bb.y+bb.height), 
        Range(bb.x, bb.x+bb.width));

    visualizer::img::show(
      "result", 
      outImg
    );


    cv::imwrite("out.png", outImg);

    
    visualizer::slider::update();
    sleep(1);
	}

	return 0;
}
