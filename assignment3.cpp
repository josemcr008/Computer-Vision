#include <iostream>
#include <exception>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


cv::Mat convolve(Mat &image, Mat &kernel)
{

     cv::Mat aux=cv::Mat(image.rows + (kernel.rows-1), image.cols + (kernel.rows-1), CV_32FC1, 0.0);
     image.copyTo(aux(cv::Rect((kernel.rows-1)/2,(kernel.cols-1)/2,image.cols,image.rows)));

     Mat result(image.rows, image.cols, CV_32FC1);
     float sum;
     int r=kernel.rows/2;

     for (int x = 0; x < image.rows; x++) {
         for (int y = 0; y < image.cols; y++) {
           sum=0;
             for (int i =0; i <  kernel.rows;i++) {
                 for (int j = 0; j < kernel.cols; j++) {
                   sum+=kernel.at<float>(i,j)*aux.at<float>(x-i+2*r, y-j+2*r);

                  }

              }
              result.at<float>(x,y)=sum;


            }

        }


    return result;
}


void applyKernel(cv::Mat &image, cv::Mat &image_out, cv::Mat &kernel)
{


  if(kernel.type()!=CV_32FC1)
  {
    std::cout << "Error, kernel must be CV_32FC1" << '\n';
  }
  image_out=convolve(image, kernel);
  //cv::imshow("final image", image_out);

}


int main(int argc,char **argv){
try{
  if(argc!=2) {cerr<<"Usage:image"<<endl;return 0;}
  //loads the image from file
  cv:: Mat image, image_out;

   image=cv::imread(argv[1]);
   if( image.rows==0) {cerr<<"Error reading image"<<endl;return 0;}
   //creates a window
   cvtColor(image, image, cv::COLOR_BGR2GRAY );
   //cvtColor(image, image_out, cv::COLOR_BGR2GRAY );

   namedWindow("final image");
   namedWindow("original image");
   image.convertTo(image, CV_32FC1, 1.0/255.0);

   cv::imshow("original image", image);

   cv::Mat kernel(3,3, CV_32FC1, 1.0/9.0);

   applyKernel(image, image_out, kernel);

    cv::imshow("final image", image_out);


   char c=0;
   cout<<"Press ESC to exit\n";
   while(c!=27)  //waits until ESC pressed
	  c=cv::waitKey(0);

}catch(std::exception &ex)
{
  cout<<ex.what()<<endl;
}

}
