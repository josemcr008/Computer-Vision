#include <iostream>
#include <string>
#include <exception>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


class CmdLineParser{
   int argc;
   char **argv;
public:
  CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}
  bool operator[] ( string param ) {int idx=-1; for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; return ( idx!=-1 ) ; }
  string operator()(string param,string defvalue="-1"){int idx=-1; for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue; else return ( argv[ idx+1] ); }
};


int calculaleMin(vector<float> v)
{
  int min=v[0];
  for(int i=0; i<v.size(); i++)
  {
    if(min>v[i])
    {
      min=v[i];
    }
  }
  return min;
}


int calculaleMax(vector<float> v)
{
  int max=0;
  for(int i=0; i<v.size(); i++)
  {
    if(max<v[i])
    {
      max=v[i];
    }
  }
  return max;
}


void applyErosion(cv::Mat &out)
{

  vector<float> v;
  float aux=0;
  int size=3;

  /*cv::Mat exp=cv::Mat(out.rows + (size-1), out.cols + (size-1), CV_32FC1, 0.0);
  out.copyTo(exp(cv::Rect((size-1)/2,(size-1)/2,out.cols,out.rows)));
*/

  int r=size/2;
  int min=0;

  for (int x = 0; x < out.rows; x++) {
      for (int y = 0; y < out.cols; y++) {
        v.clear();
          for (int i=0; i <  size; i++) {
              for (int j = 0; j < size; j++) {
                aux=out.at<float>(x-i+2*r, y-j+2*r);
                v.push_back(aux);

                }

           }
           min=calculaleMin(v);
           out.at<float>(x,y)=min;
         }
     }
}



void applyDilate(cv::Mat &out)
{

  vector<float> v;
  float aux=0;
  int size=3;

  /*cv::Mat exp=cv::Mat(out.rows + (size-1), out.cols + (size-1), CV_32FC1, 0.0);
  out.copyTo(exp(cv::Rect((size-1)/2,(size-1)/2,out.cols,out.rows)));
*/

  int r=size/2;
  int max=0;

  for (int x = 0; x < out.rows; x++) {
      for (int y = 0; y < out.cols; y++) {
        v.clear();
          for (int i=0; i <  size; i++) {
              for (int j = 0; j < size; j++) {
                aux=out.at<float>(x-i+2*r, y-j+2*r);
                v.push_back(aux);

                }

           }
           max=calculaleMax(v);
           out.at<float>(x,y)=max;
         }
     }
}



void applyThreshold(const cv::Mat &in, cv::Mat &out, float th)
{
  for (int x = 0; x < in.rows; x++) {
      for (int y = 0; y < in.cols; y++) {
                if(in.at<float>(x,y)>=th)
                {
                  out.at<float>(x,y)=0;
                }
                else{
                  out.at<float>(x,y)=255;
                }
      }
  }
}





int main(int argc,char **argv){
try{
  if(argc!=6) {cerr<<"./progam /image.jpg -thres <value> -op <erode|dilate|open|close>"<<endl;return 0;}
  //loads the image from file
  cv:: Mat image, image_out;

  CmdLineParser cml(argc,argv);

  //check if a command is present
  if (cml["-thres"]){
    float th= stof( cml("-thres"));
}
  if (cml["-op"]){
    string op=cml("-op");
}
   float th=stof(cml("-thres","100"));//if -p is not, then, return 1
   string op=cml("-op", "erode");


   image=cv::imread(argv[1]);
   if( image.rows==0) {cerr<<"Error reading image"<<endl;return 0;}


   cvtColor(image, image, cv::COLOR_BGR2GRAY, 1.0/255.0);
   image.convertTo(image, CV_32FC1);
   image.copyTo(image_out);

   applyThreshold(image, image_out, th);
   //image_out.convertTo(image_out, CV_8UC1);
   cv::namedWindow("th");
   cv::imshow("th",image_out);


   if(op=="erode")
   {
     applyErosion(image_out);
     cv::namedWindow("erode");
     cv::imshow("erode",image_out);

   }
   else if(op=="dilate")
   {
     applyDilate(image_out);
     cv::namedWindow("dilate");
     cv::imshow("dilate",image_out);
   }
   else if(op=="open")
   {
     applyErosion(image_out);
     applyDilate(image_out);
     cv::namedWindow("open");
     cv::imshow("open",image_out);
   }
   else if(op=="close")
   {
     applyDilate(image_out);
     applyErosion(image_out);
     cv::namedWindow("close");
     cv::imshow("close",image_out);
   }
   else{
     cout<<"Error, the operation are: erode, dilate, open and close\n";
     return 0;
   }


   char c=0;
   cout<<"Press ESC to exit\n";
   while(c!=27)  //waits until ESC pressed
	  c=cv::waitKey(0);

}catch(std::exception &ex)
{
  cout<<ex.what()<<endl;
}

}
