#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include <opencv2/objdetect.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

// Global variables
map<string,int> thresh = {
    {"Thresh", 110}
};

RNG rng(12345);

void trackThing();
void approxAndDraw(Mat graysrc, vector<vector<Point>> &contours, Mat &overlay);

int main(int argc, char *argv[])
{
    trackThing();
    return 0;
}

// getCircles...
void trackThing(){
    VideoCapture cap("clipped2.mp4");
    Mat orig, orig2, img_gray, img_gray2, img_thresh, shapes, final;
    vector<vector<Point>> contours;

    // Windows
    string objtrackname = "Visual";
    namedWindow(objtrackname, CV_WINDOW_AUTOSIZE);

    if(!cap.isOpened()){
        cout << "Failed to open video\n";
        exit(EXIT_FAILURE);
    }

    while(true){
        if(!cap.read(orig)){
            cout << "End of stream reached!\n";
            exit(EXIT_FAILURE);
        }

        cvtColor(orig,img_gray, COLOR_BGR2GRAY);

        if(!cap.read(orig2)){
            cout << "End of stream reached!\n";
            exit(EXIT_FAILURE);
        }
        cvtColor(orig2,img_gray2, COLOR_BGR2GRAY);

        absdiff(img_gray,img_gray2,final);

        //Blur image
        blur(final, final, Size(5,5));

        threshold(final, img_thresh, thresh["Thresh"],  255, THRESH_BINARY);
        findContours(img_thresh, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        approxAndDraw(img_thresh, contours, shapes);

        // if(final is all zeros) point = prevpoint;

        // Display image
        imshow(objtrackname, img_thresh);

        if(cv::waitKey(30) == 27){
            cap.release();
            break;
        }
    }
    cv::destroyAllWindows();
}

// approxAndDraw takes some contours and calculates and
// draws stuff...
void approxAndDraw(Mat graysrc, vector<vector<Point>> &contours, Mat &overlay){
    // Declare polygon variables
    vector<vector<Point>> contours_poly(contours.size());
    vector<Rect> boundRects(contours.size());
    vector<Point2f> center(contours.size());
    vector<float> radius(contours.size());

    // Declare moment variables
    vector<Moments> objMomentsVec(contours.size());
    vector<Point> objCentroids(contours.size());

    // Approximate contours to polygons and build bounding rectangles
    for(unsigned int i = 0; i < contours.size(); ++i){
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        boundRects[i] = boundingRect(Mat(contours_poly[i]));
        minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
    }

    // Draw bounding rectangles
    for(unsigned int i = 0; i < contours.size(); ++i){
        Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
        if(radius[i] > 15){
            drawContours(overlay, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
//            rectangle(overlay, boundRects[i].tl(), boundRects[i].br(), color, 2, 8, 0);
            objMomentsVec[i] = moments(graysrc(boundRects[i]).clone());
            if(objMomentsVec[i].m00 > 1){
                objCentroids[i] = Point(objMomentsVec[i].m10 / objMomentsVec[i].m00,
                                        objMomentsVec[i].m01 / objMomentsVec[i].m00) + boundRects[i].tl();
                circle(overlay, objCentroids[i], 26, Scalar(0,255,0),-1);
            }
        }
    }
}
