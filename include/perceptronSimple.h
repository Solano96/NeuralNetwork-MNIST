
#ifndef _PERCEPTRONSIMPLE_H_
#define _PERCEPTRONSIMPLE_H_

#include <math.h>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <sys/time.h>
#include <random>

using namespace std;

class SimplePerceptron{
    private:
        vector<double> weights;
        double bias;
        int input_size;

    public:
        SimplePerceptron(int input_size_);
        void train(vector<vector<double> > &dataset, vector<int> &labels, int epochs_ = 10, double learning_rate_ = 0.01);
        double get_output(vector<double> input);
};


class MnistSimplePerceptron{
    private:
        vector<SimplePerceptron> neurons;
        int input_size;

    public:
        MnistSimplePerceptron(int input_size_);
        void train(vector<vector<double> > &dataset, vector<int> &label, int epochs_ = 10, double learning_rate_ = 0.01);
        int predict(vector<double> &data);
        double get_accuracy(vector<vector<double> > &x, vector<int> &y);
};

#endif
