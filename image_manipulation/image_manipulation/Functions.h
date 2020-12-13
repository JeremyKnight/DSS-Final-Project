#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "RGB.h"

void d_convert_greyscale(RGB* pixels, int height, int width);

void d_convert_blur(RGB* pixels, int height, int width);

void d_edge_detection(RGB* pixels, int height, int width);

void d_laplacian(RGB* pixels, int height, int width);

void d_contrast(RGB* pixels, int height, int width, int rincrease, int gincrease, int bincrease);

void d_brightness(RGB* pixels, int height, int width, int bright);

#endif