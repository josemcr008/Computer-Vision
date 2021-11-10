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


    cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
    //image.copyTo(image_out);
    image_out=image.clone();



    const int histSize = 256;
    int *hist = new int[histSize] {};
    int *accumulatedHist_norm = new int[histSize] {};


    for (int y = 0; y < image.rows; y++)
      for (int x = 0; x < image.cols; x++)
        hist[(int)image.at<uchar>(y, x)]++;


    auto i=0;

    while(!hist[i]) i++;

    int n_rows=image.rows;
    int n_cols=image.cols;
    auto size=n_rows*n_cols;

    float aux=0;
    float scale = (histSize - 1.f) / (size - hist[i]);

    for (accumulatedHist_norm[i++] = 0; i < histSize; i++){
        aux+= hist[i];


        accumulatedHist_norm[i] = cv::saturate_cast<uchar>(aux*scale);
    }

    for (int y = 0; y < image.rows; y++)
          for (int x = 0; x < image.cols; x++)
          {
              image_out.at<uchar>(y, x) = accumulatedHist_norm[(int)image.at<uchar>(y, x)];
          }
    std::cout << "flag5" << '\n';
      // Display equilized image
    cv::namedWindow("Equilized Image");
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
