// readMNIST.cc
// read MNIST data into double vector, OpenCV Mat, or Armadillo mat
// free to use this code for any purpose
// author : Eric Yuan
// my blog: http://eric-yuan.me/
// part of this code is stolen from http://compvisionlab.wordpress.com/


#ifndef _READMNIST_H_
#define _READMNIST_H_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <armadillo>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <sys/time.h>
#include <random>

using namespace cv;
using namespace std;
using namespace arma;

int ReverseInt (int i);
void read_Mnist(string filename, vector<vector<double> > &vec);

void read_Mnist(string filename, vector<cv::Mat> &vec);

void read_Mnist(string filename, vector<arma::mat> &vec);

void read_Mnist_Label(string filename, vector<int> &vec);

void read_Mnist_Label(string filename, arma::colvec &vec);

void normalize_dataset(vector<vector<double> > &dataset);

#endif
