/**
* CS - 315 (Distributed Scalable Computing) Converting to greyscale
* */
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <chrono>
#include <mutex>
using namespace std;
using namespace std::chrono;

#include "bmp.h"

#define NUM_IMAGES	3

void d_convert_greyscale(RGB * pixels, int height, int width);
void plusBlurLauncher(RGB* pixels, int height, int width);
void squareBlurLauncher(RGB* pixels, int height, int width);

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

int main()
{
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

		//convert image using cpu plus
		auto start = high_resolution_clock::now();
		d_convert_greyscale(pixels, image.getHeight(), image.getWidth());
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		image.setImageFromRGB(pixels);
		image.saveBMP("grayScale.bmp");

		//printing out the duration for results
		cout << "plus cpu took: " << duration.count() << " microseconds to make resultPlusCPU.bmp" << endl;

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