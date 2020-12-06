#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "RGB.h"
#include <iostream>

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


// |Gx(x,y)| = -P(x-1,y-1) + -2 *P(x-1,y) + -P(x-1,y+1) + P(x+1,y-1) + 2 * P(x + 1, y) + P(x + 1, y + 1)
// |Gy(x,y)| = P(x-1,y-1) + 2*P(x,y-1) + P(x+1,y-1) + -P(x-1,y+1) + –2 * P(x, y + 1) - P(x + 1, y + 1)

__global__ void edgeDetectionKernel(RGB* d_pixels, int height, int width)
{
	// determine the current pixel
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	/*float h, s, i;
	i = (d_pixels[row * width + col].red + d_pixels[row * width + col].green + d_pixels[row * width + col].blue) / 3;
	h = acos(((d_pixels[row * width + col].red - d_pixels[row * width + col].green) + (d_pixels[row * width + col].red - d_pixels[row * width + col].blue) / (2 * sqrt(((d_pixels[row * width + col].red - d_pixels[row * width + col].green) * (d_pixels[row * width + col].red - d_pixels[row * width + col].green)) + ((d_pixels[row * width + col].red - d_pixels[row * width + col].blue) * (d_pixels[row * width + col].green - d_pixels[row * width + col].blue))))));
	if (d_pixels[row * width + col].red < d_pixels[row * width + col].green && d_pixels[row * width + col].red < d_pixels[row * width + col].blue) {
		s = d_pixels[row * width + col].red / i;
	}
	else if (d_pixels[row * width + col].green < d_pixels[row * width + col].red && d_pixels[row * width + col].green < d_pixels[row * width + col].blue) {
		s = d_pixels[row * width + col].green / i;
	}
	else {
		s = d_pixels[row * width + col].blue / i;
	}

	d_pixels[row * width + col].red = h;
	d_pixels[row * width + col].green = s;
	d_pixels[row * width + col].blue = i;*/

	float reddx, reddy;
	float greendx, greendy;
	float bluedx, bluedy;

	if (col < width && row < height) {
		if (col > 0 && row > 0 && col < width && row < height) {
			//red
			reddx = (-1 * d_pixels[(row - 1) * width + (col - 1)].red) + (-2 * d_pixels[row * width + (col - 1)].red) + (-1 * d_pixels[(row + 1) * width + (col - 1)].red) + (d_pixels[(row - 1) * width + (col + 1)].red) + (2 * d_pixels[row * width + (col + 1)].red) + (d_pixels[(row + 1) * width + (col + 1)].red);
			reddy = (d_pixels[(row - 1) * width + (col - 1)].red) + (2 * d_pixels[(row - 1) * width + col].red) + (d_pixels[(row - 1) * width + (col + 1)].red) + (-1 * d_pixels[(row + 1) * width + (col - 1)].red) + (-2 * d_pixels[(row + 1) * width + col].red) + (-1 * d_pixels[(row + 1) * width + (col + 1)].red);
			
			//reddx /= 5;
			//reddy /= 5;
			
			d_pixels[row * width + col].red = sqrt((reddx * reddx) + (reddy * reddy));
			
			//green
			greendx = (-1 * d_pixels[(row - 1) * width + (col - 1)].green) + (-2 * d_pixels[row * width + (col - 1)].green) + (-1 * d_pixels[(row + 1) * width + (col - 1)].green) + (d_pixels[(row - 1) * width + (col + 1)].green) + (2 * d_pixels[row * width + (col + 1)].green) + (d_pixels[(row + 1) * width + (col + 1)].green);
			greendy = (d_pixels[(row - 1) * width + (col - 1)].green) + (2 * d_pixels[(row - 1) * width + col].green) + (d_pixels[(row - 1) * width + (col + 1)].green) + (-1 * d_pixels[(row + 1) * width + (col - 1)].green) + (-2 * d_pixels[(row + 1) * width + col].green) + (-1 * d_pixels[(row + 1) * width + (col + 1)].green);
			
			//greendx /= 5;
			//greendy /= 5;
			
			d_pixels[row * width + col].green = sqrt((greendx * greendx) + (greendy * greendy));
			
			//blue
			bluedx = (-1 * d_pixels[(row - 1) * width + (col - 1)].blue) + (-2 * d_pixels[row * width + (col - 1)].blue) + (-1 * d_pixels[(row + 1) * width + (col - 1)].blue) + (d_pixels[(row - 1) * width + (col + 1)].blue) + (2 * d_pixels[row * width + (col + 1)].blue) + (d_pixels[(row + 1) * width + (col + 1)].blue);
			bluedy = (d_pixels[(row - 1) * width + (col - 1)].blue) + (2 * d_pixels[(row - 1) * width + col].blue) + (d_pixels[(row - 1) * width + (col + 1)].blue) + (-1 * d_pixels[(row + 1) * width + (col - 1)].blue) + (-2 * d_pixels[(row + 1) * width + col].blue) + (-1 * d_pixels[(row + 1) * width + (col + 1)].blue);
			
			//bluedx /= 5;
			//bluedy /= 5;
			
			d_pixels[row * width + col].blue = sqrt((bluedx * bluedx) + (bluedy * bluedy));
		}
	}
}

__host__ void d_edge_detection(RGB* pixel, int height, int width)
{
	RGB* d_pixel;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	edgeDetectionKernel << <grid, block >> > (d_pixel, height, width);

	cudaMemcpy(pixel, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}
