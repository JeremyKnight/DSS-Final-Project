#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "RGB.h"
#include <iostream>
#include "math.h"
#include <stdio.h>

/**
* Helper function to calculate the greyscale value based on R, G, and B
*/
__device__ int greyscale(BYTE red, BYTE green, BYTE blue)
{
	int grey = 0.3 * red + 0.59 * green + 0 * 11 * blue; // calculate grey scale
	return min(grey, 255);
}

/**
* Kernel for executing on GPY
*/
__global__ void greyscaleKernel(RGB* d_pixels, int height, int width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (y >= height || y >= width)
		return;

	int index = y * width + x;

	int grey = greyscale(d_pixels[index].red, d_pixels[index].green, d_pixels[index].blue); // calculate grey scale

	d_pixels[index].red = grey;
	d_pixels[index].green = grey;
	d_pixels[index].blue = grey;

}

/**
*	Helper function to calculate the number of blocks on an axis based on the total grid size and number of threads in that axis
*/
__host__ int calcBlockDim(int total, int num_threads)
{
	int r = total / num_threads;
	if (total % num_threads != 0) // add one to cover all the threads per block
		++r;
	return r;
}

/**
*	Host function for launching greyscale kernel
*/
__host__ void d_convert_greyscale(RGB* pixel, int height, int width)
{
	RGB* d_pixel;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	greyscaleKernel << <grid, block >> > (d_pixel, height, width);

	cudaMemcpy(pixel, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}

__global__ void plusBlurKernel(RGB* d_pixels, int height, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height
	int index = y * width + x;

	int sumRed = d_pixels[index].red;
	int sumBlue = d_pixels[index].blue;
	int sumGreen = d_pixels[index].green;
	int numOfChanges = 1;

	if (y >= height || y >= width)
		return;
	if (y + 1 < height) {
		int i = (y + 1) * width + x;
		sumRed += d_pixels[i].red;
		sumBlue += d_pixels[i].blue;
		sumGreen += d_pixels[i].green;
		numOfChanges++;
	}
	if (y - 1 > 0) {
		int i = (y - 1) * width + x;
		sumRed += d_pixels[i].red;
		sumBlue += d_pixels[i].blue;
		sumGreen += d_pixels[i].green;
		numOfChanges++;
	}
	if (x + 1 < width) {
		int i = y * width + (x + 1);
		sumRed += d_pixels[i].red;
		sumBlue += d_pixels[i].blue;
		sumGreen += d_pixels[i].green;
		numOfChanges++;
	} if (x - 1 > 0) {
		int i = y * width + (x - 1);
		sumRed += d_pixels[i].red;
		sumBlue += d_pixels[i].blue;
		sumGreen += d_pixels[i].green;
		numOfChanges++;
	}

	d_pixels[index].red = sumRed / numOfChanges;
	d_pixels[index].green = sumGreen / numOfChanges;
	d_pixels[index].blue = sumBlue / numOfChanges;
}

__host__ void plusBlurLauncher(RGB* pixel, int height, int width) {
	RGB* d_pixel;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	plusBlurKernel << <grid, block >> > (d_pixel, height, width);

	cudaMemcpy(pixel, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);

}

__global__ void  squareBlurKernel(RGB* pixels, int height, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	int index = y * width + x;

	int sumRed = pixels[index].red;
	int sumBlue = pixels[index].blue;
	int sumGreen = pixels[index].green;
	int numOfChanges = 1;

	if (y >= height || y >= width) {
		return;
	}

	//use a double for loop, and loop through the area adding into the averages while it is within bounds

	for (int i = -2; i < 2; i++) {
		for (int j = -2; j < 2; j++) {
			if ((x + i > 0 && y + j > 0) && (x + i < width && y + j < height)) {
				int is = (y + j) * width + (x + i);
				sumRed += pixels[is].red;
				sumBlue += pixels[is].blue;
				sumGreen += pixels[is].green;
				numOfChanges++;
			}

		}
	}
	pixels[index].red = sumRed / numOfChanges;
	pixels[index].green = sumGreen / numOfChanges;
	pixels[index].blue = sumBlue / numOfChanges;
}

//for edgeDir and gradiant I will have to use the index shit that has been used in all of the above stuff
__global__ void GradiantStrength(RGB* pixels, int* edgeDir, int* gradiant, int height, int width) {
	//printf("hi");
	int row = blockIdx.x * blockDim.x + threadIdx.x; // width
	int col = blockIdx.y * blockDim.y + threadIdx.y; // height
	int newAngle = 0;
	int index = col * width + row;
	int* GxMask = (int*)malloc(width*3);
	int* GyMask = (int*)malloc(width*3);
	
	//int GxMask[1000000];				// Sobel mask in the x direction
	//int GyMask[1000000];				// Sobel mask in the y direction
	//printf("hello there");
	//sobel mask set up
	GxMask[width * 4] = 20;

	GxMask[0] = -1; GxMask[1] = -2;  GxMask[2] = -1;
	GxMask[width] = 0;  GxMask[width + 1] = 0;  GxMask[width + 2] = 0;
	GxMask[width * 2] = 1;  GxMask[width * 2 + 1] = 2;  GxMask[width * 2 + 2] = 1;
	GyMask[0] = 1; GyMask[1] = 0; GyMask[2] = -1;
	GyMask[width] = 2; GyMask[width + 1] = 0; GyMask[width + 2] = -2;
	GyMask[width * 2] = 1; GyMask[width * 2 + 1] = 0; GyMask[width * 2 + 2] = -1;
	
	printf("ahhhhh");
	if (col >= height || col >= width) {
		return;
	}
	printf("a");
	//long i = (unsigned long)(row * 3 * width + 3 * col);
	double Gx = 0;
	double Gy = 0;
	/* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
	for (int rowOffset = -1; rowOffset <= 1; rowOffset++) {
		for (int colOffset = -1; colOffset <= 1; colOffset++) {
			int rowTotal = row + rowOffset;
			int colTotal = col + colOffset;
			long iOffset = (unsigned long)(rowTotal * 3 * width + colTotal * 3);

			int gIndex = colOffset * width + rowOffset + 1;
			Gx = Gx + (pixels[index].red * (GxMask[gIndex]));//the image should have already been changed to grayscale so any color should be fine
			//std::cout << "red from graidnant strength in kernel is: " << pixels[index].red << std::endl;
			printf("red from graidnant strength in kernel is: %d", pixels[index].red);
			Gy = Gy + (pixels[index].red * (GyMask[gIndex]));
		}
	}

	gradiant[index] = sqrt((Gx*Gx) + (Gy*Gy));	// Calculate gradient strength			
	
	double thisAngle = (atan2(Gx, Gy) / 3.14159) * 180.0;		// Calculate actual direction of edge

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

__host__ void squareBlurLauncher(RGB* pixel, int height, int width) {
	RGB* d_pixel;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	squareBlurKernel << <grid, block >> > (d_pixel, height, width);

	cudaMemcpy(pixel, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}

__host__ void gradiantLauncher(RGB* pixels, int* edgeDir, int* gradiant, int height, int width) {
	RGB* d_pixel;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixels, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	GradiantStrength << <grid, block >> > (d_pixel, edgeDir, gradiant, height, width);
	cudaMemcpy(pixels, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}
