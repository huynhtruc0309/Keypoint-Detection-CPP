#include "Header.h"

Mat detectHarrist(Mat img, int threshold, int k)
{
    // Filter images with Gaussian filter to reduce noise.
    Mat dXX, dYY, dXY, filerImg;
    Mat gaussKernel = getGaussianKernel(5, 0, CV_32F);
    sepFilter2D(img, filerImg, CV_32F, gaussKernel.t(), gaussKernel);

    // Calculate the direction of the derivative in the xand y directions at each pixel.
    Sobel(filerImg, dXX, CV_32F, 2, 0);
    Sobel(filerImg, dYY, CV_32F, 0, 2);
    Sobel(filerImg, dXY, CV_32F, 1, 1);
  
    //Calculate R = det M – k * (trace M) 2
    Mat dst = Mat::zeros(img.size(), CV_32FC1);
    Mat detH = dXX.mul(dYY) - dXY.mul(dXY);
    Mat traceH = dXX + dYY;
    dst = detH - k*traceH.mul(traceH);
    normalize(dst, dst, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    Mat desImg;
    cvtColor(img, desImg, COLOR_GRAY2RGB);
    //If R > T is a corner detected, retain the local extreme.
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            if ((int)dst.at<float>(i, j) > threshold)
            {
                circle(desImg, Point(j, i), 5, Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }
    return desImg;  
}

bool checkOverlap(int xA, int yA, double rA, int xB, int yB, double rB, float threshold)
{
    // rA have to bigger than rB
    float distance = sqrt(pow(xA - xB, 2) + pow(yA - yB, 2));
    
    if (distance > (rA + rB)) // No overlap
        return false;

    else if (distance <= rA - rB) // Inside
    {
        return true;
    }
    else // Overlap
    {
        float area;
        int d1, d2;

        d1 = pow(rA, 2) - pow(rB, 2) + pow(distance, 2) / (2*distance);
        d2 = distance - d1;

        area = pow(rA, 2) * acos(d1 / rA) - d1 * sqrt(pow(rA, 2) - pow(d1, 2))
            + pow(rB, 2) * acos(d2 / rB) - d2 * sqrt(pow(rB, 2) - pow(d2, 2));

        if (area >= threshold * 3.14 * pow(rB, 2) && area < 0.99 *3.14 * pow(rB, 2))
            return true;
    }

    return false;
}

Mat detectBlob(Mat img, int threshold)
{
    Mat GXX, GYY, gaussKernel, tempDst;
    float sigma = 1.0;
    Mat LoG;
    vector<Mat> dstNorm(9);
    //vector<vector<double>> keypoints (img.rows, vector<double>(img.cols, 0));
    vector<tuple<int, int, float>> keypoints;

    for (int k = 0; k < 9; k++)
    {
        gaussKernel = getGaussianKernel(5, sigma, CV_32F);
        Sobel(gaussKernel, GXX, CV_32F, 2, 0);
        Sobel(gaussKernel, GYY, CV_32F, 0, 2);
        LoG = GXX + GYY;

        sepFilter2D(img, tempDst, CV_32F, LoG.t(), LoG);
        normalize(tempDst, dstNorm[k], 0, 255, NORM_MINMAX, CV_32FC1, Mat());

        sigma = sigma * sqrt(2);
    }

    for (int x = 1; x < img.rows - 1; x++)
    {
        for (int y = 1; y < img.cols - 1; y++)
        {
            int indexMax = 0;
            int maxValue = 0;

            for (int k = 0; k < 9; k++)
            {
                for (int row = x - 1; row < x + 2; row++)
                {
                    for (int col = y - 1; col < y + 2; col++)
                    {
                        if ((int)dstNorm[k].at<float>(row, col) > maxValue)
                        {
                            maxValue = (int)dstNorm[k].at<float>(row, col);
                            indexMax = k;
                        }
                    }
                }
            }

            if (maxValue > threshold) {
                keypoints.push_back(make_tuple(x, y, pow(sqrt(2), indexMax)));
                //keypoints[x][y] = pow(sqrt(2), indexMax);
            }
        }
    }
    /*
    //Remove redundance blob
    for (int i = 0; i < keypoints.size() - 1; i++)
    {
        for (int j = i + 1; j < keypoints.size(); j++)
        {
            if (get<2>(keypoints[i]) > get<2>(keypoints[j]))
            {
                if (checkOverlap(get<0>(keypoints[i]), get<1>(keypoints[i]), get<2>(keypoints[i]), get<0>(keypoints[j]), get<1>(keypoints[j]), get<2>(keypoints[j]), 0.98))
                {
                    keypoints.erase(keypoints.begin(), keypoints.begin() + i);
                    i--;
                    break;
                }
            }
            else
            {
                if (checkOverlap(get<0>(keypoints[j]), get<1>(keypoints[j]), get<2>(keypoints[j]), get<0>(keypoints[i]), get<1>(keypoints[i]), get<2>(keypoints[i]), 0.98))
                {
                    keypoints.erase(keypoints.begin(), keypoints.begin() + j);
                    j--;
                }
            }
        }
    }
    */
    //Drawing the blobs.
    Mat desImg;
    cvtColor(img, desImg, COLOR_GRAY2RGB);
    /*
    for (int i = 0; i < keypoints.size(); i++)
    {
        for (int j = 0; j < keypoints[0].size(); j++)
        {
            if (keypoints[i][j])
                circle(desImg, Point(j, i), keypoints[i][j] * sqrt(2), Scalar(0, 0, 255), 1, 8, 0);
        }
    }
    */
    for (int i = 0; i < keypoints.size(); i++)
    {
        circle(desImg, Point(get<1>(keypoints[i]), get<0>(keypoints[i])), get<2>(keypoints[i]) * sqrt(2), Scalar(0, 0, 255), 1, 8, 0);
    }

    return desImg;
}

Mat detectDOG(Mat img) {
    Mat gaussKernel, tempDst;
    float sigma = 1.0;
    vector<Mat> dstNorm(9);
    vector<vector<double>> keypoints(img.rows, vector<double>(img.cols, 0));

    for (int k = 0; k < 9; k++)
    {
        gaussKernel = getGaussianKernel(5, sigma, CV_32F);
        sepFilter2D(img, tempDst, CV_32F, gaussKernel.t(), gaussKernel);
        normalize(tempDst, dstNorm[k], 0, 255, NORM_MINMAX, CV_32FC1, Mat());

        sigma = sigma * sqrt(2);
    }

    for (int k = 0; k < dstNorm.size() - 1; k++) 
    {
        dstNorm[k] = dstNorm[k] - dstNorm[k + 1];
    }
    dstNorm.resize(dstNorm.size() - 1);



    return img;
}

double matchBySIFT(Mat img1, Mat img2, int detector) {

    return 0;
}