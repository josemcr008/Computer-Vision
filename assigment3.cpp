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


     Mat aux=Mat(image.rows + (kernel.rows-1), image.cols + (kernel.rows-1), CV_32FC1, 0.0);
     image.copyTo(aux(Rect((kernel.rows-1)/2,(kernel.rows-1)/2,image.cols,image.rows)));

     Mat result_mult;
     Mat result;
     image.copyTo(result);

     for(int i=0; i<image.rows; i++)
     {
         for(int j=0; j<image.cols; j++)
         {
             result_mult=kernel.mul(aux(Rect(j,i,kernel.rows,kernel.cols)));
             float suma=sum(result_mult)[0];
             result.at<float>(i,j)=suma;
         }
     }
    return result;
}


void applyKernel(cv::Mat &image, cv::Mat &image_out, cv::Mat &kernel)
{


  if(kernel.type()!=CV_32FC1)
  {
    std::cout << "Error, kernel must be CV_32FC1" << '\n';
    exit(-1);
  }
  image_out=convolve(image, kernel);

  cv::imshow("final image", image_out);
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

   char c=0;
   cout<<"Press ESC to exit\n";
   while(c!=27)  //waits until ESC pressed
	  c=cv::waitKey(0);

}catch(std::exception &ex)
{
  cout<<ex.what()<<endl;
}

}
