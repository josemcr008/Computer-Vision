/**
* Jose Maria Cabrera Rivas
* i82carij@uco.es
* Assigment 1 : ForegroundHighlight
**/
#include <opencv2/core/core.hpp> //core routines
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>//imread,imshow,namedWindow,waitKey
#include <iostream>
using namespace std;
cv::Mat image;
cv::Mat image2;
cv::Mat image_bw;
cv::Point pt1, pt2;


void on_mouse( int event, int x, int y, int flags, void* param )
{

    image.copyTo(image2);

    if(event==cv::EVENT_LBUTTONDOWN){
      pt1=cv::Point(x,y);
    }
    else if(event==cv::EVENT_LBUTTONUP){
      pt2=cv::Point(x,y);
      cv::rectangle(image2, pt1, pt2, cv::Scalar(0, 0, 0),-1, cv::LINE_8);

      cv::cvtColor(image2, image_bw,cv::COLOR_BGR2GRAY);
      vector<cv::Mat> channels;

      channels.push_back(image_bw);
      channels.push_back(image_bw);
      channels.push_back(image_bw);
      merge(channels, image_bw);


      for(int y=0;y<image2.rows;y++){
	       uchar *ptr=image.ptr<uchar>(y);//pointer to the y-th image row
         uchar *ptr2=image_bw.ptr<uchar>(y);//pointer to the y-th image row
	        for(int x=0;x<image2.cols;x++){

            if(ptr2[0]==0 && ptr2[1]==0 && ptr2[2]==0)
            {
              ptr2[0]=ptr[0];
              ptr2[1]=ptr[1];
              ptr2[2]=ptr[2];
            }
		        ptr2+=3;//moves the pointer 3 elements
		        ptr+=3;//moves the pointer 3 elements
          }
      }
      cv::namedWindow("final image");
  	  cv::imshow("final image",image_bw);

  }


}



int main(int argc,char **argv){
try{
  if(argc!=2) {cerr<<"Usage:image"<<endl;return 0;}
  //loads the image from file
   image=cv::imread(argv[1]);
   if( image.rows==0) {cerr<<"Error reading image"<<endl;return 0;}
   //creates a window
   cv::namedWindow("image");
   cv::setMouseCallback( "image", on_mouse, 0 );
   //displays the image in the window
   cv::imshow("image",image);
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
