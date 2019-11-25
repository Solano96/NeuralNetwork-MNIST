
#include "activations.h"
#include <math.h>       /* exp */


double sigmoid(double x){
	return 1.0/(1.0+exp(-x));
}

vector<double> sigmoid(vector<double> &x){
	vector<double> sigmoid_vector;
	int x_size = x.size();

	for(int i = 0; i < x_size; i++){
		sigmoid_vector.push_back(1.0/(1.0+exp(-x[i])));
	}

	return sigmoid_vector
}

double sigmoid_prime(double x){
	return sigmoid(x)*(1-sigmoid(x));
}

vector<double> sigmoid_prime(vector<double> &x){
	vector<double> sigmoid_prime_vector;

	for(int i = 0; i < x_size; i++){
		sigmoid_prime_vector.push_back(sigmoid_prime(x[i]));
	}

	return sigmoid_prime_vector;

}
