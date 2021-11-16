/* 
   (c) Fundamentos de Sistemas Inteligenties en Vision
   University of Cordoba, Spain  
*/

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctype.h>
#include <cmath>


#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

#include "seglib.hpp"

using namespace std;
using namespace cv;

#ifndef NDEBUG
int __Debug_Level = 0;
#endif

/*
  Use CMake to compile
*/

const cv::String keys =
    "{help h usage ? |      | print this message   }"        
#ifndef NDEBUG
    "{verbose        |0     | Set the verbose level.}"
#endif    
    "{t threshold    |13     | Segmentation threshold.}"
    "{s              |0   | Radius of structural element.}"    
    "{c              |  | Use the camera?}"
    "{@input         |<none>| Path to video/Camera index.}"
    "{@output        |<none>| Path to output video.}"
    ;

int main (int argc, char * const argv[])
{
  /* Default values */
  bool cameraInput=false;
  int threshold;
  const char * filein = 0;
  const char * fileout = 0;
  char opt;
  int radius = 0;
  
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Background segmentation in video.");
  if (parser.has("help"))
  {
      parser.printMessage();
      return 0;
  }

#ifndef NDEBUG
  __Debug_Level = parser.get<int>("verbose");
#endif
    
  std::string input_path = parser.get<std::string>("@input");
  std::string output_path = parser.get<std::string>("@output");
  threshold = parser.get<int>("threshold");  
  radius = parser.get<int>("s");

  filein = input_path.c_str();
  fileout = output_path.c_str();

  std::cout << "Input stream: " << filein << endl;
  std::cout << "Output: " << fileout << endl;

  VideoCapture video(input_path);
    
  if (parser.has("c"))
  {
    std::cout << "Using camera index" << std::endl;
    video.open(atoi(filein));
  }
  else  
    video.open(filein);
    
  if (!video.isOpened())
  {
    cerr << "Error: the input stream is not opened.\n";
    abort();
  }

  Mat inFrame;
  bool wasOk = video.read(inFrame);
  if (!wasOk)
  {
    cerr << "Error: could not read any image from stream.\n";
    abort();
  }
  
  double fps=25.0;
  if (!cameraInput)
    fps=video.get(CV_CAP_PROP_FPS);
  std::cout << fps << std::endl;
  
  Mat outFrame = Mat::zeros(inFrame.size(), CV_8UC3);
  Mat theMask = Mat::zeros(inFrame.size(), CV_8UC1);
  
  std::cout << inFrame.size() << std::endl;
  VideoWriter output;
  
  output.open(fileout, CV_FOURCC('M','J','P','G'), fps, inFrame.size());
  if (!output.isOpened())
  {
    cerr << "Error: the ouput stream is not opened.\n";
    exit(-1);
  }  

  int frameNumber=0;
  int key = 0;

  Mat difimg, prevFrame;

  inFrame.copyTo(prevFrame);
  cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);

// WRITE ME 

  cv::namedWindow("Output");

  while(video.grab())
  {
    
    video.retrieve(inFrame);
    cvtColor(inFrame, inFrame, cv::COLOR_BGR2GRAY);

    inFrame.copyTo(difimg);


    fsiv_segm_by_dif(prevFrame, inFrame, difimg,  threshold,  radius);
    cv::imshow ("diff", difimg);  

    frameNumber++;

    inFrame.copyTo(prevFrame);


	 // Do your processing
	 // TODO


    // TODO Apply the mask to the original frame and show

    // Preparing the next iteration

    // TODO Add frame to output video
    cv::waitKey(5);
  }    
}
