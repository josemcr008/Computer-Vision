/***********************************
 * Autor: Jose Maria Cabrera Rivas
 *
 * Assigment 2: Image Equalization

 * ******************************************/
#include <opencv2/core/core.hpp> //core routines
#include <opencv2/highgui/highgui.hpp>//imread,imshow,namedWindow,waitKey
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>


using namespace std;
using namespace cv;

int main(int argc,char **argv){
try{
  if(argc!=2) {cerr<<"Usage:image"<<endl;return 0;}
  //loads the image from file
   cv::Mat image=cv::imread(argv[1]);
   if( image.rows==0) {cerr<<"Error reading image"<<endl;return 0;}
   //creates a window
   cv::namedWindow("myimage");
   //displays the image in the window
   cv::imshow("myimage",image);

    cv::Mat image_out;
    cv::Mat image_ex;


    cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
    //image.copyTo(image_out);
    image_out=image.clone();


    int hist[256] {};
    int com_hist[256] {};
    int hist_norm[256] {};

    int aux=0;


//Histogram
for (int y = 0; y < image.rows; y++)
{
  for (int x = 0; x < image.cols; x++)
  {
    hist[image.at<uchar>(y, x)]++;
  }
}

//Comulative histogram
    for(int i=0; i<256; i++)
    {
      aux += hist[i];
      com_hist[i] = aux;
    }


    float max= com_hist[255];

//Normalize histogram
     for(int i=0; i<256; i++)
     {
       hist_norm[i]=com_hist[i]*(255/max);
     }


    for(int y = 0; y<image.rows; y++)
        for(int x = 0; x<image.cols; x++)
        {
              image_out.at<uchar>(y,x)=hist_norm[image_out.at<uchar>(y,x)];
        }



    cv::imshow("Equilized Image",image_out);


   //wait for a key to be pressed
   char c=0;
   cout<<"Press ESC to exit\n";
   while(c!=27)  //waits until ESC pressed
	  c=cv::waitKey(0);


}catch(std::exception &ex)
{
  cout<<ex.what()<<endl;
}

}
