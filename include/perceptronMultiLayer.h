#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include "activations.h"
#include "utils.h"

using namespace std;

class Network{
private:
	double num_layers;
	vector<int> sizes;
	vector<vector<double> > biases;
	vector<vector<vector<double> > > weights;

	void biases_fill(vector<vector<double> > &b, int n_layers, vector<int> &l_sizes);
	void weights_fill(vector<vector<vector<double> > > &w, int n_layers, vector<int> &l_sizes, bool random_w);
	vector<double> feedForward(vector<double> &a);
	void SGD(vector<vector<double> > &x_train, vector<vector<int> > &y_train, int epochs, int mini_batch_size, double eta, vector<vector<double> > &x_test, vector<vector<int> > &y_test);
	void update_mini_batch(vector<vector<double> > &x_mini_batch, vector<vector<int> > &y_mini_batch, double eta);
	void backprop(vector<double> &x, vector<int> &y, vector<vector<double> > &nabla_b, vector<vector<vector<double> > > &nabla_w);
	vector<double> cost_derivative(vector<double> &outputs_activations, vector<int> &y);

public:
	Network(vector<int> sizes);
	void train(vector<vector<double> > &dataset, vector<int> &label, vector<vector<double> > &x_test, vector<int> &y_test);
	int predict(vector<double> &data);
};
