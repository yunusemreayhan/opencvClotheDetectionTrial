#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <set>
#include <queue>
#include <iostream>
#include <list>

#include "generic.hpp"

using namespace std;
using namespace cv;

const float RATIO_OF_SELECTION = 1.4; 
const int TOP_STOP_AFTER_PIXEL_CNT = 100;
const int ACCEPTED_COLOR_NEIGH_DISTANCE = 15;

inline bool isPointWithinBorders(Mat toCheck, Point2i point) {
    if(point.x < 0 || point.x >= toCheck.rows)
        return false;
    if(point.y < 0 || point.y >= toCheck.cols)
        return false;
    return true;
}

Vec3b convertUint32ToVec3b(const uint32_t &inp) {
    Vec3b ret;
    ret[0] = inp & 0xff;
    ret[1] = (inp >> 8) & 0xff;
    ret[2] = (inp >> 16) & 0xff;
    return ret;
}

uint32_t convertVec3bToUint32(const Vec3b &inp) {
    uint32_t ret;
    ret = inp[0] | (inp[1] << 8) | (inp[2] << 16);
    return ret;
}

inline bool checkColorCode(int code) {
    if(code >= 0 && code < 256) {
        return true;
    }
    return false;
}

int getTopBoundaryRowNormal(Mat image, set<uint32_t> &backgrounColors) {
    size_t latest_size = 50000;
    int i = 0;
    for(; i < image.rows; i++) {
        set<uint32_t> b;
        for(int j = 0; j < image.cols; j++) {
            auto & itr_label_val = image.at<Vec3b>(i, j);
            for (size_t lx = 0; lx < 3; lx++)
            {
                uint32_t toinsert = convertVec3bToUint32(itr_label_val);
                b.insert(toinsert);
            }
        }
        //cout << "row : " << i << "  latest size : " << latest_size << "  size : " << b.size()<< endl;
        if(b.size() > 100)
            return i;
        if(((float)b.size()) > (((float)latest_size) * RATIO_OF_SELECTION)) {
            return i==image.rows-1?0:i;
        } else {
            for(auto color : b) {
                backgrounColors.insert(color);
            }
        }
        latest_size = b.size();
    }
    return i==image.rows-1?0:i;
}

int getBottomBoundaryRowNormal(Mat image, set<uint32_t> &backgrounColors) {
    size_t latest_size = 50000;
    for(int i = image.rows - 1; i >= 0; i--) {
        set<uint32_t> b;
        for(int j = 0; j < image.cols; j++) {
            auto & itr_label_val = image.at<Vec3b>(i, j);
            for (size_t lx = 0; lx < 3; lx++)
            {
                uint32_t toinsert = itr_label_val[0] | itr_label_val[1] << 8 | itr_label_val[2] << 16;
                b.insert(toinsert);
            }
        }
        //cout << "row : " << i << "  latest size : " << latest_size << "  size : " << b.size()<< endl;
        if(b.size() > 100)
            return i;
        if(((float)b.size()) > (((float)latest_size) * RATIO_OF_SELECTION)) {
            return i==0?image.rows-1:i;
        } else {
            for(auto color : b) {
                backgrounColors.insert(color);
            }
        }
        latest_size = b.size();
    }
    return image.rows-1;
}

int getLeftBoundaryColNormal(Mat image, set<uint32_t> &backgrounColors) {
    size_t latest_size = 50000;
    for(int j = 0; j < image.cols; j++) {
        set<uint32_t> b;
        for(int i = 0; i < image.rows; i++) {
            auto & itr_label_val = image.at<Vec3b>(i, j);
            for (size_t lx = 0; lx < 3; lx++)
            {
                uint32_t toinsert = itr_label_val[0] | itr_label_val[1] << 8 | itr_label_val[2] << 16;
                b.insert(toinsert);
            }
        }
        // cout << "col : " << j << "  latest size : " << latest_size << "  size : " << b.size()<< endl;
        if(b.size() > 100)
            return j;
        if(((float)b.size()) > (((float)latest_size) * RATIO_OF_SELECTION)) {
            return j==image.cols-1?0:j;
        } else {
            for(auto color : b) {
                backgrounColors.insert(color);
            }
        }
        latest_size = b.size();
    }
    return 0;
}

int getRightBoundaryColNormal(Mat image, set<uint32_t> &backgrounColors) {
    size_t latest_size = 50000;
    for(int j = image.cols -1; j >= 0; j--) {
        set<uint32_t> b;
        for(int i = 0; i < image.rows; i++) {
            auto & itr_label_val = image.at<Vec3b>(i, j);
            for (size_t lx = 0; lx < 3; lx++)
            {
                uint32_t toinsert = itr_label_val[0] | itr_label_val[1] << 8 | itr_label_val[2] << 16;
                b.insert(toinsert);
            }
        }
        // cout << "col : " << j << "  latest size : " << latest_size << "  size : " << b.size()<< endl;
        if(b.size() > 100)
            return j;
        if(((float)b.size()) > (((float)latest_size) * RATIO_OF_SELECTION)) {
            return j==0?image.cols-1:j;
        } else {
            for(auto color : b) {
                backgrounColors.insert(color);
                //insertColorWithNearColors(convertUint32ToVec3b(color), backgrounColors, DISTANCE_OF_COLOR_TO_INSERT_AS_BACKGROUND);
            }
        }
        latest_size = b.size();
    }
    return image.cols - 1;
}

void getBorders(Mat input, list<Point2i> &ret) {
    // insert left
    for(int i = 0; i < input.rows; i++) {
        Point2i toinsert{i, 0};
        ret.push_back(toinsert);
    }
    // insert top
    for(int j = 0; j < input.cols; j++) {
        Point2i toinsert{0, j};
        ret.push_back(toinsert);
    }
    // insert right
    for(int i = 0; i < input.rows; i++) {
        Point2i toinsert{i, input.cols - 1 };
        ret.push_back(toinsert);
    }
    // insert top
    for(int j = 0; j < input.cols; j++) {
        Point2i toinsert{input.rows - 1, j};
        ret.push_back(toinsert);
    }
}


inline bool isPixelVisited(const Point2i &coordinate, Mat mask) {
    if(mask.at<uint8_t>(coordinate.x, coordinate.y) == 255)
        return true;
    return false;
}



inline bool isColorMatchInNeighbors(const Vec3b &colorToCheck, const set<uint32_t> &neighborColors, int accepted_distance) {
    for(auto color: neighborColors) {
        if(norm(convertUint32ToVec3b(color), colorToCheck) < accepted_distance) {
            return true;
        }
    }
    return false;
}

void getNeighbors(const Mat toBfs, const Point2i toCheck, list<Point2i> &neighs, const set<uint32_t> &neighborColors, Mat mask, int accepted_distance) {
    const int neighDistance = 1;
    for(int i = toCheck.x - neighDistance ; i <= toCheck.x + neighDistance; i++) {
        for(int j = toCheck.y - neighDistance ; j <= toCheck.y + neighDistance; j++) {
            const Vec3b &pixelToCheck = toBfs.at<Vec3b>(i, j);
            if(isPointWithinBorders(toBfs, {i, j}) && 
                    !isPixelVisited({i, j}, mask) && 
                    isColorMatchInNeighbors(pixelToCheck, neighborColors, accepted_distance)) {
                neighs.push_back({i, j});
            }
        }
    }
}

void bfs(Mat toBfs, const Point2i startPos, const set<uint32_t> &neighborColors, Mat mask, int accepted_distance) {
    queue<Point2i> m_queue;

    if(isColorMatchInNeighbors(toBfs.at<Vec3b>(startPos.x, startPos.y), neighborColors, accepted_distance)) {
        m_queue.push(startPos);
    } else {
        return;
    }
    while(!m_queue.empty()) {
        list<Point2i> neighs;
        getNeighbors(toBfs, m_queue.front(), neighs, neighborColors, mask, accepted_distance);
        m_queue.pop();
        for(auto neigh : neighs) {
            auto &pixel = mask.at<uint8_t>(neigh.x, neigh.y);
            if(!(pixel == 255)) {
                pixel = 255;
                m_queue.push(neigh);
            }
        }
    }
}

void startBfsFromCorners(Mat toBfs, const set<uint32_t> &neighborColors, Mat mask, int accepted_color_distance) {
    list<Point2i> borders;
    getBorders(toBfs, borders);
    for(Point2i border : borders) {
        bfs(toBfs, border, neighborColors, mask, accepted_color_distance);
    }
}

Mat getMaskForImage(Mat input) {
    return {input.size(), CV_8UC1, Scalar(0,0,0)};
}

Mat startBfsForEachPixel(Mat toBfs, const set<uint32_t> &neighborColors, Mat mask, int accepted_color_distance) {
    Mat ret;
    toBfs.copyTo(ret);
    for(int i = 0; i < toBfs.rows; i++) {
        for(int j = 0; j < toBfs.cols; j++) {
            bfs(ret, {i, j}, neighborColors, mask, accepted_color_distance);
        }
    }
    return ret;
}

set<uint32_t> getSkinColors() {
    set<uint32_t> skincolors {
        convertVec3bToUint32({36,85,141}),
        convertVec3bToUint32({224,172,105}),
        convertVec3bToUint32({225,164,137}),
        convertVec3bToUint32({241,194,125}),
        convertVec3bToUint32({255,219,172}),
        convertVec3bToUint32({66,134,197})};
    return skincolors;
}

Mat removeSkinColor(Mat input, Mat mask, int accepted_color_distance) {
    return startBfsForEachPixel(input, getSkinColors(), mask, accepted_color_distance);
}

list<Vec3b> getBackgroundColors() {
    Vec3b backgroundFillColor {0, 255, 0};
    list<Vec3b> fillColors{backgroundFillColor};
    return fillColors;
}

cv::Rect cropImage(Mat input, set<uint32_t> &backgrounColors) {
    size_t top, left, right, bottom;
    top = getTopBoundaryRowNormal(input, backgrounColors);
    bottom = getBottomBoundaryRowNormal(input, backgrounColors);
    left = getLeftBoundaryColNormal(input, backgrounColors);
    right = getRightBoundaryColNormal(input, backgrounColors);
    cout << "size of background colors : " << backgrounColors.size() << endl;
    cout << "top cut : " << top << endl;
    cout << "bottom cut : " << input.rows - bottom << endl;
    cout << "left cut : " << left << endl;
    cout << "right cut : " << input.cols - right << endl;
    cv::Rect crop_region(left, top, input.cols - left - 1 - (input.cols - right), input.rows - top - 1 - (input.rows - bottom));
    return crop_region;
}

Mat getBackgroundMask(Mat input, set<uint32_t> &backgrounColors) {    
    Mat mask;
    Mat lapret = laplacianImage(input);
    mask = getMaskForImage(lapret);

    startBfsFromCorners(lapret, backgrounColors, mask, 3);

    medianBlur(mask, mask, 5);
    // GaussianBlur(mask, mask, {31,31}, 0);
    auto kernel = getStructuringElement(MORPH_ELLIPSE, {5,5});
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    bitwise_not(mask, mask);
    
    return mask;
}

Mat removeBackground(Mat input, Mat mask) {
    Mat out;
    bitwise_and(input, input, out, mask);
    return out;
}