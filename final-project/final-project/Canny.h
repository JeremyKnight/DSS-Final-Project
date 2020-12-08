#ifndef _Canny_H_
#define _Canny_H_

//#include "stdafx.h"
//#include "tripod.h"
//#include "tripodDlg.h"

//#include "LVServerDefs.h"
#include "math.h"
#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "kernel.cu"

using namespace std;
class Canny {
private:
	RGB* pixels;
	int width;
	int height;
public:
	
	Canny(RGB* p, int w, int h) {
		pixels = p;
		width = w;
		height = h;
	}

	~Canny() {
		delete pixels;
	}

	
	/*
	void gaussaianBlur()
	{
		// gaussaianBlur:  This is where you'd write your own image processing code
		// Task: Read a pixel's grayscale value and process accordingly


		unsigned int W, H;		// Width and Height of current frame [pixels]
		unsigned int row, col;		// Pixel's row and col positions
		unsigned long i;		// Dummy variable for row-column vector
		int upperThreshold = 60;	// Gradient strength nessicary to start edge
		int lowerThreshold = 30;	// Minimum gradient strength to continue edge
		unsigned long iOffset;		// Variable to offset row-column vector during sobel mask
		int rowOffset;			// Row offset from the current pixel
		int colOffset;			// Col offset from the current pixel
		int rowTotal = 0;		// Row position of offset pixel
		int colTotal = 0;		// Col position of offset pixel
		int Gx;				// Sum of Sobel mask products values in the x direction
		int Gy;				// Sum of Sobel mask products values in the y direction
		float thisAngle;		// Gradient direction based on Gx and Gy
		int newAngle;			// Approximation of the gradient direction
		bool edgeEnd;			// Stores whether or not the edge is at the edge of the possible image
		int GxMask[3][3];		// Sobel mask in the x direction
		int GyMask[3][3];		// Sobel mask in the y direction
		int newPixel;			// Sum pixel values for gaussian
		int gaussianMask[5][5];		// Gaussian mask

		W = lpThisBitmapInfoHeader->biWidth;  // biWidth: number of columns
		H = lpThisBitmapInfoHeader->biHeight; // biHeight: number of rows

		for (row = 0; row < H; row++) {
			for (col = 0; col < W; col++) {
				edgeDir[row][col] = 0;
			}
		}

		//Declare Sobel masks
		GxMask[0][0] = -1; GxMask[0][1] = 0; GxMask[0][2] = 1;
		GxMask[1][0] = -2; GxMask[1][1] = 0; GxMask[1][2] = 2;
		GxMask[2][0] = -1; GxMask[2][1] = 0; GxMask[2][2] = 1;

		GyMask[0][0] = 1; GyMask[0][1] = 2; GyMask[0][2] = 1;
		GyMask[1][0] = 0; GyMask[1][1] = 0; GyMask[1][2] = 0;
		GyMask[2][0] = -1; GyMask[2][1] = -2; GyMask[2][2] = -1;

		Declare Gaussian mask
		gaussianMask[0][0] = 2;	 gaussianMask[0][1] = 4;  gaussianMask[0][2] = 5;  gaussianMask[0][3] = 4;  gaussianMask[0][4] = 2;
		gaussianMask[1][0] = 4;	 gaussianMask[1][1] = 9;  gaussianMask[1][2] = 12; gaussianMask[1][3] = 9;  gaussianMask[1][4] = 4;
		gaussianMask[2][0] = 5;	 gaussianMask[2][1] = 12; gaussianMask[2][2] = 15; gaussianMask[2][3] = 12; gaussianMask[2][4] = 2;
		gaussianMask[3][0] = 4;	 gaussianMask[3][1] = 9;  gaussianMask[3][2] = 12; gaussianMask[3][3] = 9;  gaussianMask[3][4] = 4;
		gaussianMask[4][0] = 2;	 gaussianMask[4][1] = 4;  gaussianMask[4][2] = 5;  gaussianMask[4][3] = 4;  gaussianMask[4][4] = 2;


		Gaussian Blur 
		for (row = 2; row < H - 2; row++) {
			for (col = 2; col < W - 2; col++) {
				newPixel = 0;
				for (rowOffset = -2; rowOffset <= 2; rowOffset++) {
					for (colOffset = -2; colOffset <= 2; colOffset++) {
						rowTotal = row + rowOffset;
						colTotal = col + colOffset;
						iOffset = (unsigned long)(rowTotal * 3 * W + colTotal * 3);
						newPixel += (*(m_destinationBmp + iOffset)) * gaussianMask[2 + rowOffset][2 + colOffset];
					}
				}
				i = (unsigned long)(row * 3 * W + col * 3);
				*(m_destinationBmp + i) = newPixel / 159;
			}
		}
	}
	*/
	

};



#endif