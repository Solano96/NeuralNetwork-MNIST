#include <vector>
#include <math.h>

using namespace std;

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


template <typename T>
vector<T> softmax(const vector<T> &x){
	unsigned int x_size = x.size();
	vector<T> result(x_size);
	T sum = 0.0;

	int max = 0;;

	for(unsigned int i = 1;  i < x_size; i++){
		if(x[max] < x[i]){
			max = i;
		}
	}

	for(unsigned int i = 0; i < x_size; i++){
		sum += exp(x[i]-x[max]);
	}

	for(unsigned i = 0; i < x_size; i++){
		result[i] = exp(x[i]-x[max])/sum;
	}

	return result;
}

/*
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
*/
