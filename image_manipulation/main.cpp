#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <chrono>
using namespace std;

#include "bmp.h"

#define NUM_IMAGES	4
#define EDGE_FILTERS 6

void d_convert_greyscale(RGB* pixels, int height, int width);

void d_convert_blur(RGB* pixels, int height, int width);

void d_edge_detection(RGB* pixels, int height, int width);

void d_laplacian(RGB* pixels, int height, int width);

void d_contrast(RGB* pixels, int height, int width, int rincrease, int gincrease, int bincrease);

void d_brightness(RGB* pixels, int height, int width, int bright);

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

int main()
{

	cout << "\n********* Filter Calculation Program *********" << endl;
	cout << "Written by Tyler Gamlem, Jeremy Knight, and Mason Caird" << endl << endl;

	//BitMap image("lena.bmp"); // Load the bitmap image into the BitMap object
	//RGB* pixels = image.getRGBImageArray(); // get the image array of RGB (Red, Green, and Blue) components
	////d_convert_greyscale(pixels, image.getHeight(), image.getWidth()); // kernel
	//d_edge_detection(pixels, image.getHeight(), image.getWidth()); // kernel
	//image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
	//image.saveBMP("result.bmp"); // Save bmp

	while (true) {

		string image_archive[NUM_IMAGES] = { "lena.bmp", "marbles.bmp", "sierra_02.bmp", "tiger.bmp" };
		string edge_array[EDGE_FILTERS] = { "Grey Scale", "Sobel", "lapacain", "constrast", "brightness", "blur" };

		cout << "Select an image: \n";
		for (int i = 0; i < NUM_IMAGES; ++i)
			cout << i << ": " << image_archive[i] << endl;
		cout << NUM_IMAGES << ": exit\n";

		int choice;
		int filter;
		do {
			cout << "Please choice: ";
			cin >> choice;
			if (choice == NUM_IMAGES) {
				cout << "Goodbye!\n";
				exit(0);
			}
		} while (choice < 0 || choice > NUM_IMAGES);

		// Filter Selection

		if (choice >= 0) {

			cout << "\nSelect a filter: \n";

			for (int i = 0; i < EDGE_FILTERS; ++i)
				cout << i << ": " << edge_array[i] << endl;

			cout << EDGE_FILTERS << ": Back\n";

			do {
				cout << "Filter Choice: ";
				cout << endl;
				cin >> filter;
				if (filter == EDGE_FILTERS) {
					main();
				}
			} while (filter < 0 || filter > EDGE_FILTERS);
		}

		BitMap image(image_archive[choice]); // Load the bitmap image into the BitMap object

		// Display some of the image's properties
		cout << "Image properties\n";
		cout << setw(15) << left << "Dimensions: " << image.getHeight() << " by " << image.getWidth() << endl;
		cout << setw(15) << left << "Size: " << image.getImageSize() << " bytes\n";
		cout << setw(15) << left << "Bit encoding: " << image.getBitCount() << " bits\n\n";

		RGB* pixels = image.getRGBImageArray(); // get the image array of RGB (Red, Green, and Blue) components

		auto start = std::chrono::high_resolution_clock::now(); // start timer
		auto stop = std::chrono::high_resolution_clock::now(); // stop timer
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
		int value = 0; // user Input
		string rgb_types[3] = { "Red", "Green", "Blue" }; // RGB Types

		switch (filter) {

			// greyscale (pixels, height, width)
		case 0:

			start = std::chrono::high_resolution_clock::now(); // start timer
			d_convert_greyscale(pixels, image.getHeight(), image.getWidth()); // kernel
			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("greyscale.bmp"); // Save bmp
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Greyscale Time: " << duration.count() << " milliseconds" << endl;
			break;

			// edgedetection (pixels, height, width)
		case 1:

			start = std::chrono::high_resolution_clock::now(); // start timer
			d_edge_detection(pixels, image.getHeight(), image.getWidth()); // kernel
			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("Sobel.bmp"); // Save bmp
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Sobel Time: " << duration.count() << " milliseconds" << endl;
			break;

			// laplacain (pixels, height, width)
		case 2:

			start = std::chrono::high_resolution_clock::now(); // start timer
			d_laplacian(pixels, image.getHeight(), image.getWidth()); // kernel
			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("Laplacian.bmp"); // Save bmp
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Laplacian Time: " << duration.count() << " milliseconds" << endl;
			break;

			// contrast (pexels, height, width, red increase, int green increase, blue increase)
		case 3:

			int rgb_values[3];
			cout << "\n********* RGB value *********" << endl;

			// pick color
			for (int i = 0; i < 3; i++) {
				cout << rgb_types[i] << " Value (-255 to 255): ";
				cin >> value;
				rgb_values[i] = value;
			}

			// Display Colors
			for (int i = 0; i < 3; i++) {
				cout << rgb_types[i] << ": " << rgb_values[i];
			}

			start = std::chrono::high_resolution_clock::now(); // start timer
			d_contrast(pixels, image.getHeight(), image.getWidth(), rgb_values[0], rgb_values[1], rgb_values[2]); // kernel
			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("Contrast.bmp"); // Save bmp
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Contrast Time: " << duration.count() << " milliseconds" << endl;
			break;

			// brightness (pixels, height, width, bright);
		case 4:

			do {
				cout << "******* Brightness Multiplier *******";
				cout << "brightness: 2 - 5";
				cin >> choice;
				if (choice == NUM_IMAGES) {
					cout << "Goodbye!\n";
					exit(0);
				}
			} while (choice < 2 || choice > EDGE_FILTERS);

			start = std::chrono::high_resolution_clock::now(); // start timer
			d_brightness(pixels, image.getHeight(), image.getWidth(), choice); // kernel
			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("brightness.bmp"); // Save bmp
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Brightness Time: " << duration.count() << " milliseconds" << endl;

			break;

		case 5:
			start = std::chrono::high_resolution_clock::now(); // start timer
			d_convert_blur(pixels, image.getHeight(), image.getWidth()); // kernel
			stop = std::chrono::high_resolution_clock::now(); // stop timer
			image.setImageFromRGB(pixels); // Assign the modified pixels back to the image
			image.saveBMP("Blur.bmp"); // Save bmp
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); // stop - start
			cout << "Blur Time: " << duration.count() << " milliseconds" << endl;
			break;

		default:
			// Somehow go back
			break;

		}

		cout << "Check out the results\n\n";
		char response = 'y';
		cout << "Do you wish to repeat? [y/n] ";
		cin >> response;
		if (response != 'y') {
			cout << "Sorry to see you go ...\n";
			exit(0);
		}
	}
}

