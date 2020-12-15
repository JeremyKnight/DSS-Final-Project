/**
* CS - 315 (Distributed Scalable Computing) Converting to greyscale
* */
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <stdlib.h>
using namespace std;
using namespace std::chrono;

#include "bmp.h"

#define NUM_IMAGES	3

void d_convert_greyscale(RGB * pixels, int height, int width);
void plusBlurLauncher(RGB* pixels, int height, int width);
void squareBlurLauncher(RGB* pixels, int height, int width);
void gradiantLauncher(RGB* pixels, int* edgeDir, int* gradiant, int height, int width);

void convert_greyscale(RGB* pixels, int height, int width)
{
	for (int y = 0; y < height; ++y) { // for each row in image
		for (int x = 0; x < width; ++x) { // for each pixel in the row
			int index = y * width + x; // compute index position of (y,x) coordinate

			int grey = 0.3 * pixels[index].red + 0.59 * pixels[index].green + 0 * 11 * pixels[index].blue; // calculate grey scale

			pixels[index].red = min(grey, 255);
			pixels[index].green = min(grey, 255);
			pixels[index].blue = min(grey, 255);
		}
	}
}

/**
*  Computes the average of the red, green, and blue components of an image
*
* @param pixels  The array of RGB (Red, Green, Blue) components of each pixel in the image
* @param height  The height of the image
* @param width   The width of the image
*/
void compute_component_average(RGB* pixels, int height, int width)
{
	double total_red = 0, total_green = 0, total_blue = 0;

	for (int y = 0; y < height; ++y) { // for each row in image
		for (int x = 0; x < width; ++x) { // for each pixel in the row
			int index = y * width + x; // compute index position of (y,x) coordinate
			total_red += pixels[index].red; // add the red value at this pixel to total
			total_green += pixels[index].green; // add the green value at this pixel to total
			total_blue += pixels[index].blue; // add the blue value at this pixel to total
		}
	}

	cout << "Red average: " << total_red / (height * width) << endl;
	cout << "Green average: " << total_green / (height * width) << endl;
	cout << "Blue average: " << total_blue / (height * width) << endl;

}

/**
*  blurs the image using the plus stencil
*
* @param pixels  The array of RGB (Red, Green, Blue) components of each pixel in the image
* @param height  The height of the image
* @param width   The width of the image
*/
void convertPlusBlur(RGB* pixels, int height, int width) {

	for (int y = 0; y < height; ++y) { // for each row in image
		for (int x = 0; x < width; ++x) { // for each pixel in the row
			int index = y * width + x; // compute index position of (y,x) coordinate

			int sumRed = pixels[index].red;
			int sumBlue = pixels[index].blue;
			int sumGreen = pixels[index].green;
			int numOfChanges = 1;

			if (y + 1 < height) {
				int i = (y + 1) * width + x;
				sumRed += pixels[i].red;
				sumBlue += pixels[i].blue;
				sumGreen += pixels[i].green;
				numOfChanges++;
			}
			if (y - 1 > 0) {
				int i = (y - 1) * width + x;
				sumRed += pixels[i].red;
				sumBlue += pixels[i].blue;
				sumGreen += pixels[i].green;
				numOfChanges++;
			}
			if (x + 1 < width) {
				int i = y * width + (x + 1);
				sumRed += pixels[i].red;
				sumBlue += pixels[i].blue;
				sumGreen += pixels[i].green;
				numOfChanges++;
			} if (x - 1 > 0) {
				int i = y * width + (x - 1);
				sumRed += pixels[i].red;
				sumBlue += pixels[i].blue;
				sumGreen += pixels[i].green;
				numOfChanges++;
			}

			pixels[index].red = sumRed / numOfChanges;
			pixels[index].blue = sumBlue / numOfChanges;
			pixels[index].green = sumGreen / numOfChanges;
		}
	}
}

void averageSquare(int* arr, int n, int x, int y) {
	int upper = 2;
	int lower = -upper;

	for (int i = lower; i <= upper; i++) {
		for (int j = lower; j <= upper; j++) {

		}
	}
}

/**
*  blurs the image using the square stencil
*
* @param pixels		The array of RGB (Red, Green, Blue) components of each pixel in the image
* @param height		The height of the image
* @param width		The width of the image
* @param squareSize The size of the quare (for now, it must be 5)
*/
void convertSquareBlur(RGB* pixels, int height, int width) {

	for (int y = 0; y < height; ++y) { // for each row in image
		for (int x = 0; x < width; ++x) { // for each pixel in the row
			int index = y * width + x; // compute index position of (y,x) coordinate

			int sumRed = pixels[index].red;
			int sumBlue = pixels[index].blue;
			int sumGreen = pixels[index].green;
			int numOfChanges = 1;

			//use a double for loop, and loop through the area adding into the averages while it is within bounds
			for (int i = -2; i < 2; i++) {
				for (int j = -2; j < 2; j++) {
					if ((x + i > 0 && y + j > 0) && (x + i < width && y + j < height)) {
						//cout << "x is: " << i << "y is: " << j << endl;
						int is = (y + j) * width + (x + i);
						sumRed += pixels[is].red;
						sumBlue += pixels[is].blue;
						sumGreen += pixels[is].green;
						numOfChanges++;
					}

				}
			}

			pixels[index].red = sumRed / numOfChanges;
			pixels[index].blue = sumBlue / numOfChanges;
			pixels[index].green = sumGreen / numOfChanges;
		}
	}
}

void gradiantMaker(RGB* d_pixels, int* edgeDir, int* gradiant, int height, int width) {
			int newAngle = 2000;
			int GxMask[9];				// Sobel mask in the x direction
			int GyMask[9];				// Sobel mask in the y direction
			//printf("hello there");
			//sobel mask set up
			//GxMask[width * 4] = 20;

			GxMask[0] = -1; GxMask[1] = -2;  GxMask[2] = -1;
			GxMask[3] = 0;  GxMask[4] = 0;  GxMask[5] = 0;
			GxMask[6] = 1;  GxMask[7] = 2;  GxMask[8] = 1;
			
			GyMask[0] = 1; GyMask[1] = 0; GyMask[2] = -1;
			GyMask[3] = 2; GyMask[4] = 0; GyMask[5] = -2;
			GyMask[6] = 1; GyMask[7] = 0; GyMask[8] = -1;
			
			/*
			printf("ahhhhh");
			if (col >= height || col >= width) {
				return;
			}
			*/
			//long i = (unsigned long)(row * 3 * width + 3 * col);
	//some where within this code, there is a bug in it, through which the height is some how becoming equal to the width.
	//This honestly doesn't make any sense and I have spent 3+ hours trying to find where this is originating from, but still no luck.
	for (int row = 1; row < height-1; row++) {
		for (int col = 1; col < width-1; col++) {
			int index = row * width + col;
			long Gx = 0;
			long Gy = 0;
			/* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
			if (col < width && row < height) {
				if (col > 0 && row > 0 && col < width && row < height) {
					//red
					Gx = (-1 * d_pixels[(row - 1) * width + (col - 1)].red) + (-2 * d_pixels[row * width + (col - 1)].red) + (-1 * d_pixels[(row + 1) * width + (col - 1)].red) + (d_pixels[(row - 1) * width + (col + 1)].red) + (2 * d_pixels[row * width + (col + 1)].red) + (d_pixels[(row + 1) * width + (col + 1)].red);
					Gy = (d_pixels[(row - 1) * width + (col - 1)].red) + (2 * d_pixels[(row - 1) * width + col].red) + (d_pixels[(row - 1) * width + (col + 1)].red) + (-1 * d_pixels[(row + 1) * width + (col - 1)].red) + (-2 * d_pixels[(row + 1) * width + col].red) + (-1 * d_pixels[(row + 1) * width + (col + 1)].red);


				}
			}
			/*
			for (int rowOffset = -1; rowOffset < 2; rowOffset++) {
				for (int colOffset = -1; colOffset < 2; colOffset++) {
					int rowTotal = row + rowOffset+1;
					int colTotal = col + colOffset+1;

					if (rowTotal+1 < height && colTotal+1 < width) { //&& rowTotal+1>0 && colTotal+1>0) {
						int gIndex = (rowOffset+1) * 3 + colOffset+1;
						int i = (rowTotal) * width + colTotal;
						//std::cout << gIndex << std::endl;
						//int iOffset = rowTotal * 3 * width + colTotal * 3;
						//std::cout << "iOffset: " <
						/*
						if (rowOffset == 0 && colOffset == 0) {
							std::cout << "row total: " << rowTotal << " col total: " << colTotal << std::endl;
						}
						
						int tempGThing = GxMask[gIndex];
						Gx = Gx + ((pixels[i].red + 0) * tempGThing); //the image should have already been changed to grayscale so any color should be fine
						//std::cout << "red from graidnant strength in kernel is: " << pixels[index].red << std::endl;
						//printf("red from graidnant strength in kernel is: %d", pixels[index].red);
						//std::cout << "gIndex: " << gIndex << std::endl;
						//std::cout << "rowOffset: " << rowOffset << " colOffset: " << colOffset << std::endl;
						Gy = Gy + ((pixels[i].red + 0) * (GyMask[gIndex]));
					}
					else {
						//std::cout << "row total: " << rowTotal << " col total: " << colTotal << std::endl;
						//std::cout << "height: " << height << " width: " << width <<  std::endl;
					}
				}
			}
			*/
			gradiant[index] = sqrt((Gx * Gx) + (Gy * Gy));	// Calculate gradient strength						
			double thisAngle = (atan2(Gx, Gy) / 3.14159) * 180.0;		// Calculate actual direction of edge
			//std::cout << "this Angle is: " << thisAngle << " gradiant: " << sqrt((Gx * Gx) + (Gy * Gy)) << std::endl;
			/* Convert actual edge direction to approximate value */
			if (((thisAngle < 22.5) && (thisAngle > -22.5)) || (thisAngle > 157.5) || (thisAngle < -157.5))
				newAngle = 0;
			if (((thisAngle > 22.5) && (thisAngle < 67.5)) || ((thisAngle < -112.5) && (thisAngle > -157.5)))
				newAngle = 45;
			if (((thisAngle > 67.5) && (thisAngle < 112.5)) || ((thisAngle < -67.5) && (thisAngle > -112.5)))
				newAngle = 90;
			if (((thisAngle > 112.5) && (thisAngle < 157.5)) || ((thisAngle < -22.5) && (thisAngle > -67.5)))
				newAngle = 135;

			edgeDir[index] = newAngle;		// Store the approximate edge direction of each pixel in one array
		}
	}
}


void findEdge(RGB* pixels, int* edgeDir, int* gradient, int rowShift, int colShift, int row, int col, int dir, int lowerThreshold, int width, int height) {
	int newRow = 0;
	int newCol = 0;
	unsigned long i;
	bool edgeEnd = false;

	int index = col * width + row;

	/* Find the row and column values for the next possible pixel on the edge */
	if (colShift < 0) {
		if (col > 0)
			newCol = col + colShift;
		else
			edgeEnd = true;
	}
	else if (col < width - 1) {
		newCol = col + colShift;
	}
	else 
		edgeEnd = true;		// If the next pixel would be off image, don't do the while loop
	if (rowShift < 0) {
		if (row > 0)
			newRow = row + rowShift;
		else
			edgeEnd = true;
	}
	else if (row < height - 1) {
		newRow = row + rowShift;
	}
	else 
		edgeEnd = true;
	
	//std::cout << "new row: " << newRow << " newCol: " << newCol << std::endl;
	/* Determine edge directions and gradient strengths */
	int newIndex = newCol * width + newRow;
	while ((edgeDir[newIndex] == dir) && !edgeEnd && (gradient[newIndex] > lowerThreshold)) {
		/* Set the new pixel as white to show it is an edge */
		//i = (unsigned long)(newRow * 3 * width + 3 * newCol);
		/*
		pixels[newIndex].red + i = 255;
		*pixels[newIndex].blue + i + 1 = 255;
		*pixels[newIndex].green + i + 2 = 255;
		*/
		pixels[newIndex].red = 255;
		pixels[newIndex].blue = 255;
		pixels[newIndex].green = 255;

		if (colShift < 0) {
			if (newCol > 0)
				newCol = newCol + colShift;
			else
				edgeEnd = true;
		}
		else if (newCol < width - 1) {
			newCol = newCol + colShift;
		}
		else
			edgeEnd = true;
		if (rowShift < 0) {
			if (newRow > 0)
				newRow = newRow + rowShift;
			else
				edgeEnd = true;
		}
		else if (newRow < height - 1) {
			newRow = newRow + rowShift;
		}
		else
			edgeEnd = true;
		
		newIndex = newCol * width + newRow;
	}
}

void traceEdge(RGB* pixels, int* edgeDir, int* gradient, int height, int width) {
	/* Trace along all the edges in the image */
	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			bool edgeEnd = false;
			int index = r * width + c;
			//std::cout << gradient[index] << std::endl;
			if (gradient[index] > 60) {		//this went over // Check to see if current pixel has a high enough gradient strength to be part of an edge
				/* Switch based on current pixel's edge direction */
				switch (edgeDir[index]) {
				case 0:
					findEdge(pixels,edgeDir, gradient,0, 1, r, c, 0, 30, width, height);
					break;
				case 45:
					findEdge(pixels, edgeDir, gradient, 1, 1, r, c, 45, 30, width,height);
					break;
				case 90:
					findEdge(pixels, edgeDir, gradient, 1, 0, r, c, 90, 30, width, height);
					break;
				case 135:
					findEdge(pixels, edgeDir, gradient, 1, -1, r, c, 135, 30, width, height);
					break;
				default:
					//i = (unsigned long)(row * 3 * W + 3 * col);
					pixels[index].red = 0;
					pixels[index].blue = 0;
					pixels[index].green = 0;
					break;
				}
			}
			else {
				pixels[index].red = 0;
				pixels[index].blue = 0;
				pixels[index].green = 0;
			}
		}
	}

	/* Suppress any pixels not changed by the edge tracing */
	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			// Recall each pixel is composed of 3 bytes
			//i = (unsigned long)(row * 3 * width + 3 * col);
			int index = r * width + c;
			// If a pixel's grayValue is not black or white make it black
			if (((pixels[index].red != 255) && (pixels[index].green != 0)) || ((pixels[index].blue != 255))) { // && (*(m_destinationBmp + i + 1) != 0)) || ((*(m_destinationBmp + i + 2) != 255))) {
				//&& (*(m_destinationBmp + i + 2) != 0)))
				pixels[index].red = 0;
				pixels[index].blue = 0;
				pixels[index].green = 0;
			}
		}
	}

	/*
	//Non-maximum Suppression 
	for (row = 1; row < H - 1; row++) {
		for (col = 1; col < W - 1; col++) {
			i = (unsigned long)(row * 3 * W + 3 * col);
			if (*(m_destinationBmp + i) == 255) {		// Check to see if current pixel is an edge
				//Switch based on current pixel's edge direction 
				switch (edgeDir[row][col]) {
				case 0:
					suppressNonMax(1, 0, row, col, 0, lowerThreshold);
					break;
				case 45:
					suppressNonMax(1, -1, row, col, 45, lowerThreshold);
					break;
				case 90:
					suppressNonMax(0, 1, row, col, 90, lowerThreshold);
					break;
				case 135:
					suppressNonMax(1, 1, row, col, 135, lowerThreshold);
					break;
				default:
					break;
				}
			}
		}
		
	}
	*/
}

int main() {
	do {
		string image_archive[NUM_IMAGES] = { "lena.bmp", "marbles.bmp", "sierra_02.bmp" };
		cout << "Select an image: \n";
		for (int i = 0; i < NUM_IMAGES; ++i)
			cout << i << ": " << image_archive[i] << endl;
		cout << NUM_IMAGES << ": exit\n";

		int choice;
		do {
			cout << "Please choice: ";
			cin >> choice;
			if (choice == NUM_IMAGES) {
				cout << "Goodbye!\n";
				exit(0);
			}
		} while (choice < 0 || choice > NUM_IMAGES);

		BitMap image(image_archive[choice]); // Load the bitmap image into the BitMap object

		// Display some of the image's properties
		cout << "Image properties\n";
		cout << setw(15) << left << "Dimensions: " << image.getHeight() << " by " << image.getWidth() << endl;
		cout << setw(15) << left << "Size: " << image.getImageSize() << " bytes\n";
		cout << setw(15) << left << "Bit encoding: " << image.getBitCount() << " bits\n\n";

		RGB* pixels = image.getRGBImageArray(); // get the image array of RGB (Red, Green, and Blue) components
		int height = image.getHeight();
		int width = image.getWidth();
		//const int ISize = width * height;
		std::cout << width * height << std::endl;
		int* gradiant = (int*)malloc((height * width) * 300);
		int* edgeDir = (int*)malloc((height * width) * 300);

		/*
		int GxMask[9];				// Sobel mask in the x direction
		int GyMask[9];

		GxMask[0] = -1; GxMask[1] = -2;  GxMask[2] = -1;
		GxMask[3] = 0;  GxMask[4] = 0;  GxMask[5] = 0;
		GxMask[6] = 1;  GxMask[7] = 2;  GxMask[8] = 1;

		GyMask[0] = 1; GyMask[1] = 0; GyMask[2] = -1;
		GyMask[3] = 2; GyMask[4] = 0; GyMask[5] = -2;
		GyMask[6] = 1; GyMask[7] = 0; GyMask[8] = -1;
		*/

		//turn to grey
		d_convert_greyscale(pixels, image.getHeight(), image.getWidth()); //convert image using grayScale
		//convert_greyscale(pixels, image.getHeight(), image.getWidth());
		image.setImageFromRGB(pixels);
		image.saveBMP("grayScale.bmp");

		
		//turn to blur
		//plusBlurLauncher(pixels, image.getHeight(), image.getWidth());
		//convertSquareBlur(pixels, image.getHeight(), image.getWidth());
		//image.setImageFromRGB(pixels);
		//image.saveBMP("PlusBlurDone.bmp");
		

		//find gradiant or alter gradiant
		gradiantLauncher(pixels, edgeDir, gradiant, image.getHeight(), image.getWidth());
		//gradiantMaker(pixels, edgeDir, gradiant, height, width);
		image.setImageFromRGB(pixels);
		image.saveBMP("Gradiant.bmp");
		//for this, I can either pass Canny.h as either a holder object to hold the info so that it get passed within each object
		//or I can use it as a helper function to help with

		//RGB* pixels, int* edgeDir, int* gradient, int rowShift, int colShift, int row, int col, int dir, int lowerThreshold, int width, int height
		traceEdge(pixels, edgeDir, gradiant, image.getHeight(), image.getWidth());
		image.setImageFromRGB(pixels);
		image.saveBMP("WTF.bmp");

		/*
		//convert image using cpu plus
		auto start = high_resolution_clock::now();
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		//printing out the duration for results
		cout << "plus cpu took: " << duration.count() << " microseconds to make resultPlusCPU.bmp" << endl;
		*/

		free(gradiant);
		free(edgeDir);
		cout << "Check out test.bmp (click on it) to see image processing result\n\n";
		char response = 'y';
		cout << "Do you wish to repeat? [y/n] ";
		cin >> response;
		if (response != 'y') {
			cout << "Sorry to see you go ...\n";
			exit(0);
		}
	} while (true);
}