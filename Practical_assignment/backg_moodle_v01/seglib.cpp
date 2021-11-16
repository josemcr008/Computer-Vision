// ----------------------------------------
// seglib.cpp
// (c) FSIV, University of Cordoba
// ----------------------------------------

#include "seglib.hpp"

using namespace cv;
using namespace std;

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
  int r=size/2;
  int min=0;

  for (int x = 0; x < out.rows; x++) {
      for (int y = 0; y < out.cols; y++) {
        v.clear();
          for (int i=0; i <  size; i++) {
              for (int j = 0; j < size; j++) {
                aux=out.at<uchar>(x-i+2*r, y-j+2*r);
                v.push_back(aux);

                }

           }
           min=calculaleMin(v);
           out.at<uchar>(x,y)=min;
         }
     }
}



void applyDilate(cv::Mat &out)
{

  vector<float> v;
  float aux=0;
  int size=3;
  int r=size/2;
  int max=0;

  for (int x = 0; x < out.rows; x++) {
      for (int y = 0; y < out.cols; y++) {
        v.clear();
          for (int i=0; i <  size; i++) {
              for (int j = 0; j < size; j++) {
                aux=out.at<uchar>(x-i+2*r, y-j+2*r);
                v.push_back(aux);

                }

           }
           max=calculaleMax(v);
           out.at<uchar>(x,y)=max;
         }
     }
}


void fsiv_segm_by_dif(const cv::Mat & prevFrame, const cv::Mat & curFrame, cv::Mat & difimg, int thr, int r)
{
    absdiff(curFrame, prevFrame, difimg);
    
      for (int x = 0; x < curFrame.rows; x++) {
        for (int y = 0; y < curFrame.cols; y++) {
            if(difimg.at<uchar>(x,y)>=thr)
            {
                difimg.at<uchar>(x,y)=0;
            }
            else{
                difimg.at<uchar>(x,y)=255;
            }
        }
    }

    //Opening
    applyErosion(difimg);
    applyDilate(difimg);

    //Closing
    //applyDilate(difimg);
    //applyErosion(difimg);

}


void fsiv_apply_mask(const cv::Mat & frame, const cv::Mat & mask, cv::Mat & outframe)
{
   // WRITE ME
}

// ================= OPTIONAL PART STARTS HERE =======================

void fsiv_learn_model(cv::VideoCapture & input, int maxframes, cv::Mat & MeanI, cv::Mat & I2, cv::Mat & VarI, cv::Mat & StdI)
{
   // WRITE ME

}

void fsiv_acc_model(const cv::Mat & frame, cv::Mat & MeanI, cv::Mat & I2)
{
   // WRITE ME
}

void fsiv_segm_by_model(const cv::Mat & frame, cv::Mat & theMask, const cv::Mat & mean, const cv::Mat & std, float t, int r)
{
   // WRITE ME
}

void fsiv_update_model(const cv::Mat & frame, cv::Mat & mean, cv::Mat & I2, cv::Mat &std,  float alpha, const cv::Mat & theMask)
{
   // WRITE ME
}
