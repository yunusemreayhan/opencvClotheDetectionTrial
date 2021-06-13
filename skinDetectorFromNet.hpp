#pragma once

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

Mat getYcbrMask(Mat input) {
    auto lower = Vec3b {0, 135, 85};
    auto upper = Vec3b {255,180,135};
    //converting from gbr to YCbCr color space
    Mat img_YCrCb;
    cvtColor(input, img_YCrCb, COLOR_BGR2YCrCb);
    //skin color range for hsv color space 
    Mat YCrCb_mask;
    inRange(img_YCrCb, lower, upper, YCrCb_mask);
    auto kernel = getStructuringElement(MORPH_ELLIPSE, {4,4});
    morphologyEx(YCrCb_mask, YCrCb_mask, MORPH_OPEN, kernel);
    GaussianBlur(YCrCb_mask, YCrCb_mask, {3,3}, 0);
    bitwise_not(YCrCb_mask, YCrCb_mask);
    return YCrCb_mask;
}

Mat getHsvMask(Mat input) {
    
    auto lower = Vec3b {0, 15, 0};
    auto upper = Vec3b {17,170,255};

    Mat ret;
    cvtColor(input, ret, COLOR_BGR2HSV);
    Mat mask;
    inRange(ret, lower, upper, mask);
    auto kernel = getStructuringElement(MORPH_ELLIPSE, {4,4});
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    GaussianBlur(mask, mask, {3,3}, 0);
    bitwise_not(mask, mask);
    return mask;
}

Mat getSkinMask(Mat input) {
    Mat ret;
    Mat combined_mask;
    bitwise_and(getHsvMask(input), getYcbrMask(input), combined_mask);
    medianBlur(combined_mask, combined_mask, 3);
    auto kernel = getStructuringElement(MORPH_ELLIPSE, {4,4});
    morphologyEx(
        combined_mask,
        combined_mask,
        MORPH_OPEN,
        kernel
    );
    return combined_mask;
}

Mat detectSkin(Mat input, Mat skinMask = Mat{}) {
    if(skinMask.empty())
        skinMask = getSkinMask(input);

    Mat ret;
    bitwise_and(input, input, ret, skinMask);
    return ret;
}