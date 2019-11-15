
#ifndef _PERCEPTRONSIMPLE_H_
#define _PERCEPTRONSIMPLE_H_

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

class SimplePerceptron{
    private:
        vector<double> weights;
        double bias;
        int input_size;
        int epochs;
        double learning_rate;

    public:
        SimplePerceptron(int input_size_, int epochs_ = 10, double learning_rate_ = 0.01);
        void train(vector<vector<double> > &dataset, vector<int> &labels);
        double get_output(vector<double> input);
};


class MnistSimplePerceptron{
    private:
        vector<SimplePerceptron> neurons;
        int input_size;

    public:
        MnistSimplePerceptron(int input_size_);
        void train(vector<vector<double> > &dataset, vector<int> &label);
        int predict(vector<double> &data);
};

#endif
