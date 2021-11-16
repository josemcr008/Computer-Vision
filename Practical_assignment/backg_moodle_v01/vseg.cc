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



int main(int argc,char **argv)
{

    CmdLineParser cml(argc,argv);

    if (cml["-input"]){
        string pathToVideo=cml("-input");
    }
    if (cml["-output"]){
        string outputVideo=cml("-output");

    }
    if (cml["-t"]){
        float threshold=stof(cml("-t"));

    }

    if (cml["-s"]){
        int radius=stoi(cml("-s"));

        if(radius<0)
        {
            cout<<"s can't be < 0. s set to 0."<<endl;
            radius=0;
        }
    }

    if (cml["-g"]){
        int gauss=stoi(cml("-g"));

        if(gauss<0)
        {
            cout<<"g can't be < 0. g set to 0."<<endl;
            gauss=0;
        }
    }

    string pathToVideo= cml("-input", "../lab-3.avi");
    string outputVideo= cml("-output", "../out.avi");
    float threshold= stof(cml("-t","100"));
    int radius=stoi(cml("-s", "0"));
    int gauss=stoi(cml("-g", "0"));

   cv::VideoCapture video(pathToVideo);

   cv::Mat image, image_aux, image_out;
   while(video.grab()){

       video.retrieve(image);

        image.copyTo(image_aux);
        cvtColor(image_aux, image_aux, cv::COLOR_BGR2GRAY, 1.0/255.0);
        image_aux.convertTo(image_aux, CV_32FC1);
        image_aux.copyTo(image_out);

        applyThreshold(image_aux, image_out, threshold);

       cv::imshow("image",image);
       cv::imshow("image_aux",image_out);

       cv::waitKey(10);
   }

   

}
