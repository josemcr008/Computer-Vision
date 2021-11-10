/**
* Jose Maria Cabrera Rivas
* i82carij@uco.es
* Assigment 3 : Box Filter
**/
#include <opencv2/core/core.hpp> //core routines
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>//imread,imshow,namedWindow,waitKey
#include <iostream>
#include <cmath>


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
  else{
    image_out=convolve(image, kernel);
  }


}


int main(int argc,char **argv){
try{
  if(argc!=2) {cerr<<"Usage:image"<<endl;return 0;}
  //loads the image from file
  cv:: Mat image, image_out, image_dx, image_dy;

   image=cv::imread(argv[1]);
   if( image.rows==0) {cerr<<"Error reading image"<<endl;return 0;}
   //creates a window

   cvtColor(image, image, cv::COLOR_BGR2GRAY );
   image.convertTo(image, CV_32FC1, 1.0/255.0);
   image.copyTo(image_out);

   cv::namedWindow("original image");
   cv::imshow("original image", image);


   float sobel_x[9]={-1,0,1,-2,0,2,-1,0,1};
  cv::Mat kernel_dx(3,3, CV_32FC1, sobel_x);
  applyKernel(image, image_dx, kernel_dx);
  cv::namedWindow("dx image");
  cv::imshow("dx image", image_dx);


  float sobel_y[9]={-1,-2,-1,0,0,0,1,2,1};
 cv::Mat kernel_dy(3,3, CV_32FC1, sobel_y);
 applyKernel(image, image_dy, kernel_dy);
 cv::namedWindow("dy image");
 cv::imshow("dy image", image_dy);

 float aux_dx=0;
 float aux_dy=0;

 for(int x=0; x<image.rows; x++){
   for(int y=0; y<image.cols;y++)
   {
     aux_dx=image_dx.at<float>(x,y)*image_dx.at<float>(x,y);
     aux_dy=image_dy.at<float>(x,y)*image_dy.at<float>(x,y);
     image_out.at<float>(x,y)=sqrt(aux_dx+aux_dy);
   }
 }

 cv::normalize(image_out, image_out, 0, 255, cv::NORM_MINMAX,CV_8UC1);

 cv::namedWindow("magnitude image");
 cv::imshow("magnitude image", image_out);

   char c=0;
   cout<<"Press ESC to exit\n";
   while(c!=27)  //waits until ESC pressed
	  c=cv::waitKey(0);

}catch(std::exception &ex)
{
  cout<<ex.what()<<endl;
}

}
