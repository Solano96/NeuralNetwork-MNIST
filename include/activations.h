#ifndef _ACTIVATIONS_H_
#define _ACTIVATIONS_H_

#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;


/****************************** SIGMOID ******************************/

template <typename T>
T sigmoid(const T &x){
	return 1.0/(1.0+exp(-x));
}

template <typename T>
vector<T> sigmoid(const vector<T> &x){
	vector<T> sigmoid_vector;
	int x_size = x.size();

	for(int i = 0; i < x_size; i++){
		sigmoid_vector.push_back(sigmoid(x[i]));
	}

	return sigmoid_vector;
}

template <typename T>
T sigmoid_prime(const T &x){
	return sigmoid(x)*(1-sigmoid(x));
}

template <typename T>
vector<T> sigmoid_prime(const vector<T> &x){
	vector<T> sigmoid_prime_vector;
	int x_size = x.size();

	for(int i = 0; i < x_size; i++){
		sigmoid_prime_vector.push_back(sigmoid_prime(x[i]));
	}

	return sigmoid_prime_vector;
}

/****************************** RELU ******************************/

template <typename T>
T relu(const T &x){
	return (0.0 < x) ? x : 0.0;
}

template <typename T>
vector<T> relu(const vector<T> &x){
	vector<T> relu_vector;
	int x_size = x.size();

	for(int i = 0; i < x_size; i++){
		relu_vector.push_back(relu(x[i]));
	}

	return relu_vector;
}

template <typename T>
T relu_prime(const T &x){
	return (0.0 < x) ? 1.0 : 0.0;
}

template <typename T>
vector<T> relu_prime(const vector<T> &x){
	vector<T> relu_prime_vector;
	int x_size = x.size();

	for(int i = 0; i < x_size; i++){
		relu_prime_vector.push_back(relu_prime(x[i]));
	}

	return relu_prime_vector;
}

/****************************** SOFTMAX ******************************/

template <typename T>
vector<T> softmax(const vector<T> &x){
	unsigned int x_size = x.size();
	vector<T> result(x_size);

	T x_max = *max_element(x.begin(), x.end());
	T sum = 0.0;

	for(unsigned int i = 0; i < x_size; i++){
		sum += exp(x[i]-x_max);
	}

	for(unsigned i = 0; i < x_size; i++){
		result[i] = exp(x[i]-x_max)/sum;
	}

	return result;
}

template <typename T>
vector<T> softmax_prime(const vector<T> &x){
	unsigned int x_size = x.size();
	vector<T> result(x_size);
	vector<T> softmax_vector = softmax(x);

	for(int i = 0; i < x_size; i++){
		result[i] = softmax_vector[i]*(1-softmax_vector[i]);
	}

	return result;
}

#endif
