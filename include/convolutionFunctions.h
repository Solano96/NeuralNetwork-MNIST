#ifndef _CONVOLUTIONFUNCTIONS_H_
#define _CONVOLUTIONFUNCTIONS_H_

#include <vector>
#include <cmath>
#include <assert.h>
using namespace std;


/**
  * @brief Convert a vector into a mask (square matrix form)
  * @param x Vector to be transformed
  * @return x in square matrix form
  */
vector<vector<double> > vectorToMask(vector<double> &x);

/**
  * @brief Apply a convolution mask over x
  * @param x Vector to be convolutioned
  * @param h Hight of x
  * @param w Width of x
  * @param mask Convolution masks
  * @param msize Side size of the convolution mask
  * @return Result of convolution
  */
vector<double> convolution(vector<double> &x, int h, int w, vector<vector<double> > mask, int msize);

/**
  * @brief Pooling over x
  * @param x Vector to be convolutioned
  * @param h Hight of x
  * @param w Width of x
  * @param psize Side size for the pooling block
  */
vector<double> pooling(vector<double> &x, int h, int w, int psize);

#endif
