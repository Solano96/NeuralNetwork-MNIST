#include "activations.h"
#include <math.h>


double sigmoid(const double &x){
	return 1.0/(1.0+exp(-x));
	// ReLu
	//return ((x <= 0) ? 0 : x);
}

vector<double> sigmoid(const vector<double> &x){
	vector<double> sigmoid_vector;
	int x_size = x.size();

	for(int i = 0; i < x_size; i++){
		sigmoid_vector.push_back(sigmoid(x[i]));
	}

	return sigmoid_vector;
}

double sigmoid_prime(const double &x){
	return sigmoid(x)*(1-sigmoid(x));
	// ReLu Prime
	//return ((x <= 0) ? 0 : 1);
}

vector<double> sigmoid_prime(const vector<double> &x){
	vector<double> sigmoid_prime_vector;
	int x_size = x.size();

	for(int i = 0; i < x_size; i++){
		sigmoid_prime_vector.push_back(sigmoid_prime(x[i]));
	}

	return sigmoid_prime_vector;
}
