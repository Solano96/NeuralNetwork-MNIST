#ifndef _CONVOLUTIONALNEURALNETWORK_H_
#define _CONVOLUTIONALNEURALNETWORK_H_

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include "activations.h"
#include "convolutionFunctions.h"

using namespace std;

class CNN{
private:
	int num_layers;
	int mask_size;
	int num_masks;
	vector<int> sizes;
	vector<vector<double> > biases;
	vector<vector<vector<double> > > weights;
	vector<vector<vector<double> > > masks;

	/**
	  * @brief Fill convolution mask
	  * @param mask Reference to the mask to be filled
	  * @param mask_size Size of the side of the mask
	  */
	void mask_fill(vector<vector<double> > &mask, int mask_size, default_random_engine& generator);

	/**
	  * @brief Fill biases
	  * @param b Reference to the biases to be filled
	  * @param n_layers Numbers of layers
	  * @param l_sizes Vector containing the number of nodes in each layer
	  */
	void biases_fill(vector<vector<double> > &b, int n_layers, vector<int> &l_sizes);

	/**
	  * @brief Fill weights
	  * @param mask Reference to the biases to be filled
	  * @param n_layers Numbers of layers
	  * @param l_sizes Vector containing the number of nodes in each layer
	  * @param random_w If true weights are initialized randomly, if false are initialized to zero
	  */
	void weights_fill(vector<vector<vector<double> > > &w, int n_layers, vector<int> &l_sizes, bool random_w);

	/*
	 * @brief Get gradient for convolution mask
	 * @param x Mask input
	 * @param delta_a Mask output derivatives
	 * @param h x hight
	 * @param w x width
	 * @return Gradient of convolution mask
	 */
	vector<vector<double> > deltaConvolution(vector<double> &x, vector<double> &delta_a, int h, int w);

	/*
	 * @brief Propagate input to get the network output
	 * @param input Input of the neural network
	 * @return Output of the neural network
	 */
	vector<double> feedForward(vector<double> &input);

	/*
	 * @brief Algorithm stochastic gradient descent
	 * @param x_train Train dataset
	 * @param y_train Labels of train dataset
	 * @param x_test Test dataset
	 * @param y_test Labels of test dataset
	 * @param epochs Numbers of epochs for the SGD
	 * @param mini_batch_size Number of elements for each mini batch
	 * @param eta Learning rate value
	 */
	void SGD(vector<vector<double> > &x_train, vector<vector<int> > &y_train, vector<vector<double> > &x_test, vector<vector<int> > &y_test, int epochs, int mini_batch_size, double eta);

	/*
	 * @brief Apply backpropagation for each element of the mini batch
	 * @param x_mini_batch Mini batch with elements of the training dataset
	 * @param y_mini_batch Mini batch with labels of x_mini_batch
	 * @param eta Learning rate
	 */
	void update_mini_batch(vector<vector<double> > &x_mini_batch, vector<vector<int> > &y_mini_batch, double eta);
	void backprop(vector<double> &x, vector<int> &y, vector<vector<double> > &nabla_b, vector<vector<vector<double> > > &nabla_w, vector<vector<vector<double> > > &nabla_mask);
	vector<double> cost_derivative(vector<double> &outputs_activations, vector<int> &y);
	double loss_function(vector<vector<double> > &x, vector<vector<int> > &y);

public:
	/**
	  * @brief Network Constructor
	  * @param sizes Vector containing the number of nodes in each layer
	  */
	CNN(vector<int> sizes);
	void train(vector<vector<double> > &dataset, vector<int> &label, vector<vector<double> > &x_test, vector<int> &y_test, int epochs, int mini_batch_size, double eta);
	int predict(vector<double> &data);
	double get_accuracy(vector<vector<double> > &x, vector<vector<int> > &y);
	double get_accuracy(vector<vector<double> > &x, vector<int> &y);
};

#endif
