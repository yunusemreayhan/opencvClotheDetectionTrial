#pragma once

// Include required header files from OpenCV directory
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
  
using namespace std;
using namespace cv;
  
// Function for Face Detection
void detectAndDraw( Mat& img, CascadeClassifier& cascade, 
                CascadeClassifier& nestedCascade, double scale );
string cascadeName, nestedCascadeName;
  
int something(  )
{
    // VideoCapture class for playing video for which faces to be detected
    VideoCapture capture; 
    Mat frame, image;
  
    // PreDefined trained XML classifiers with facial features
    CascadeClassifier cascade, nestedCascade; 
    double scale=1;
  
    // Load classifiers from "opencv/data/haarcascades" directory 
    nestedCascade.load( "../../haarcascade_eye_tree_eyeglasses.xml" ) ;
  
    // Change path before execution 
    cascade.load( "../../haarcascade_frontalcatface.xml" ) ; 
  
    // Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video
    capture.open(0); 
    if( capture.isOpened() )
    {
        // Capture frames from video and detect faces
        cout << "Face Detection Started...." << endl;
        while(1)
        {
            capture >> frame;
            if( frame.empty() )
                break;
            Mat frame1 = frame.clone();
            detectAndDraw( frame1, cascade, nestedCascade, scale ); 
            char c = (char)waitKey(10);
          
            // Press q to exit from window
            if( c == 27 || c == 'q' || c == 'Q' ) 
                break;
        }
    }
    else
        cout<<"Could not Open Camera";
    return 0;
}
  
void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale)
{
    vector<Rect> faces, faces2;
    Mat gray, smallImg;
  
    cvtColor( img, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale
    double fx = 1 / scale;
  
    // Resize the Grayscale Image 
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR ); 
    equalizeHist( smallImg, smallImg );
  
    // Detect faces of different sizes using cascade classifier 
    cascade.detectMultiScale( smallImg, faces, 1.1, 
                            2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
  
    // Draw circles around the faces
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = Scalar(255, 0, 0); // Color for Drawing tool
        int radius;
  
        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cv::Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                    cv::Point(cvRound((r.x + r.width-1)*scale), 
                    cvRound((r.y + r.height-1)*scale)), color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );
          
        // Detection of eyes int the input image
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects, 1.1, 2,
                                        0|CASCADE_SCALE_IMAGE, Size(30, 30) ); 
          
        // Draw circles around eyes
        for ( size_t j = 0; j < nestedObjects.size(); j++ ) 
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }
  
    // Show Processed Image with detected faces
    imshow( "Face Detection", img ); 
}

Mat drawBodies(Mat img, vector<Rect> bodies, Scalar color) {
    Mat ret;
    img.copyTo(ret);
    double scale = 1;
    for ( size_t i = 0; i < bodies.size(); i++ )
    {
        Rect r = bodies[i];
        Point center;
        int radius;
  
        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( ret, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( ret, cv::Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                    cv::Point(cvRound((r.x + r.width-1)*scale), 
                    cvRound((r.y + r.height-1)*scale)), color, 3, 8, 0);
    }
    return ret;
}
/*
/usr/share/man/man1/opencv_haartraining.1.gz
/usr/share/opencv4/haarcascades
/usr/share/opencv4/haarcascades/haarcascade_eye.xml
/usr/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml
/usr/share/opencv4/haarcascades/haarcascade_frontalcatface.xml
/usr/share/opencv4/haarcascades/haarcascade_frontalcatface_extended.xml
/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml
/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml
/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt_tree.xml
/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml
/usr/share/opencv4/haarcascades/haarcascade_fullbody.xml
/usr/share/opencv4/haarcascades/haarcascade_lefteye_2splits.xml
/usr/share/opencv4/haarcascades/haarcascade_licence_plate_rus_16stages.xml
/usr/share/opencv4/haarcascades/haarcascade_lowerbody.xml
/usr/share/opencv4/haarcascades/haarcascade_profileface.xml
/usr/share/opencv4/haarcascades/haarcascade_righteye_2splits.xml
/usr/share/opencv4/haarcascades/haarcascade_russian_plate_number.xml
/usr/share/opencv4/haarcascades/haarcascade_smile.xml
/usr/share/opencv4/haarcascades/haarcascade_upperbody.xml
*/
void detectBodies(Mat input) {
    vector<Rect> fullbodies;
    vector<Rect> upperbodies;
    vector<Rect> lowerbodies;
    vector<Rect> faces;
    CascadeClassifier cascade_full;
    CascadeClassifier cascade_lower;
    CascadeClassifier cascade_upper;
    CascadeClassifier cascade_face;
    cascade_full.load( "/usr/share/opencv4/haarcascades/haarcascade_fullbody.xml" ) ;
    cascade_lower.load( "/usr/share/opencv4/haarcascades/haarcascade_lowerbody.xml" ) ;
    cascade_upper.load( "/usr/share/opencv4/haarcascades/haarcascade_upperbody.xml" ) ;
    cascade_face.load( "/usr/share/opencv4/haarcascades/haarcascade_frontalcatface.xml" ) ;
    Mat gray, smallImg;
    double scale=1;
    cvtColor( input, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale
    double fx = 1 / scale;
  
    // Resize the Grayscale Image 
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR ); 
    equalizeHist( smallImg, smallImg );
  
    // Detect faces of different sizes using cascade classifier
    double scaleFactor = 1.7;
    int minNeightbors = 0;
    int flags = 0|CASCADE_SCALE_IMAGE;
    Size minSize = Size(input.cols / 5, input.cols / 5); 

    cascade_full.detectMultiScale(  smallImg, fullbodies,  scaleFactor, minNeightbors, flags, minSize );
    cascade_lower.detectMultiScale( smallImg, lowerbodies, scaleFactor, minNeightbors, flags, minSize );
    cascade_upper.detectMultiScale( smallImg, upperbodies, scaleFactor, minNeightbors, flags, minSize );
    cascade_face.detectMultiScale(  smallImg, faces,       scaleFactor, minNeightbors, flags, minSize );

    //drawBodies(input, fullbodies, {255, 0, 0});
    //drawBodies(input, upperbodies, {0, 255, 0});
    //drawBodies(input, lowerbodies, {0, 0, 255});
    drawBodies(input, faces, {0, 255, 255});

    cout << endl;
}

double scaleFactor = 1.01;
inline Size minsize(Mat input) {
    return Size(input.cols / 5, input.cols / 5); 
}
inline Size maxSize(Mat input) {
    return Size(input.cols / 1.5, input.cols / 1.5); 
}
int minNeightbors = 3;
int flags = 0|CASCADE_SCALE_IMAGE;

Mat drawHead(Mat input) {
    vector<Rect> faces;
    CascadeClassifier cascade_face;
    cascade_face.load( "/usr/share/opencv4/haarcascades/haarcascade_frontalcatface.xml" ) ;
    Mat gray, smallImg;
    double scale=1;
    cvtColor( input, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale
    double fx = 1 / scale;
  
    // Resize the Grayscale Image 
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR ); 
    equalizeHist( smallImg, smallImg );
  
    // Detect faces of different sizes using cascade classifier

    cascade_face.detectMultiScale(  smallImg, faces,       scaleFactor, minNeightbors, flags, minsize(input), maxSize(input));
    return drawBodies(input, faces, {0, 255, 255});
}

Mat removeHead(Mat input) {
    vector<Rect> faces;
    CascadeClassifier cascade_face;
    cascade_face.load( "/usr/share/opencv4/haarcascades/haarcascade_frontalcatface.xml" ) ;
    Mat gray, smallImg;
    double scale=1;
    cvtColor( input, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale
    double fx = 1 / scale;
  
    // Resize the Grayscale Image 
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR ); 
    equalizeHist( smallImg, smallImg );
  
    // Detect faces of different sizes using cascade classifier


    cascade_face.detectMultiScale(  smallImg, faces,       scaleFactor, minNeightbors, flags, minsize(input), maxSize(input) );

    for(auto face : faces) {
        int topcut = face.y + face.height * 1.1;
        cv::Rect crop_region(0/*left*/, topcut/*top*/, input.cols-1/*w*/, input.rows - topcut/*h*/);
        input = input(crop_region);
        break;
    }
    return input;
}