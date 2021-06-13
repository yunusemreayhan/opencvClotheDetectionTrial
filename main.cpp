#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <set>
#include "backgroundRemover.hpp"
#include "bodyDetector.hpp"
#include "skinDetectorFromNet.hpp"
#include "DisplayImage.hpp"

#include <iostream>

using namespace std;
using namespace cv;

int main() {
    // Load the image
    for (size_t i = 1; i < 15; i++)
    {
        stringstream ss;
        ss << "inputs/lcw_trial" << i << ".jpg" ;
        Mat srcx = imread( ss.str() );
        if(srcx.empty())
            continue;
        stringstream ss0;
        ss0 << "finalresults/a_orig_image" << i << ".png" ;
        imwrite(ss0.str(), srcx);

        //equalizeHist( srcx, srcx );
        Mat kmeans2;
        srcx.copyTo(kmeans2);
        kmeans2 = getKmeansClustorImage(kmeans2, 2);
        stringstream ss8;
        ss8 << "finalresults/kmeans2_" << i << ".png" ;
        imwrite(ss8.str(), kmeans2);

        Mat lapimg;
        srcx.copyTo(lapimg);
        lapimg = laplacianImage(lapimg);
        stringstream ss9;
        ss9 << "finalresults/laplacian_img" << i << ".png" ;
        imwrite(ss9.str(), lapimg);
        
        set<uint32_t> backgroundColors;
        cv:Rect crop_region = cropImage(srcx, backgroundColors);
        Mat cropped_image = srcx(crop_region);
        Mat backgroundMask = getBackgroundMask(cropped_image, backgroundColors);
        stringstream ss3;
        ss3 << "finalresults/cropped_image" << i << ".png" ;
        imwrite(ss3.str(), cropped_image);

        Mat canny_output;
        Mat src_gray;
        int thresh = 30;
        cvtColor(lapimg, src_gray, COLOR_BGR2GRAY);
        Canny( src_gray, canny_output, thresh, thresh*2 );
        auto kernel = getStructuringElement(MORPH_ELLIPSE, {1,1});
        morphologyEx(
            canny_output,
            canny_output,
            MORPH_ERODE,
            kernel
        );
        stringstream ss10;
        ss10 << "finalresults/canny" << i << ".png" ;
        imwrite(ss10.str(), canny_output);

        Mat skinMask = getSkinMask(cropped_image);
        Mat backRemovedSkinRemovedNet = detectSkin(cropped_image, skinMask);
        stringstream ss7;
        ss7 << "finalresults/skinFromNet" << i << ".png" ;
        imwrite(ss7.str(), backRemovedSkinRemovedNet);

        Mat backRemoved = removeBackground(cropped_image, backgroundMask);
        stringstream ss5;
        ss5 << "finalresults/background_removed_image" << i << ".png" ;
        imwrite(ss5.str(), backRemoved);

        Mat combined_mask;
        Mat skin_and_background_removed;
        bitwise_and(skinMask, backgroundMask, combined_mask);
        kernel = getStructuringElement(MORPH_ELLIPSE, {15,15});
        morphologyEx(
            combined_mask,
            combined_mask,
            MORPH_OPEN,
            kernel
        );
        bitwise_and(cropped_image, cropped_image, skin_and_background_removed, combined_mask);
        stringstream ss6;
        ss6 << "finalresults/skin_and_background_removed" << i << ".png" ;
        imwrite(ss6.str(), skin_and_background_removed);

        Mat headDrawed = drawHead(cropped_image);
        stringstream ss4;
        ss4 << "finalresults/head_draw" << i << ".png" ;
        imwrite(ss4.str(), headDrawed);

        Mat headRemoved = removeHead(cropped_image);
        stringstream ss2;
        ss2 << "finalresults/head_remove" << i << ".png" ;
        imwrite(ss2.str(), headRemoved);
    }    
}