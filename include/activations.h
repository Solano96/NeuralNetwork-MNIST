#include <vector>
#include <math.h>

using namespace std;

template <typename T>
T sigmoid(const T &x){
	return 1.0/(1.0+exp(-x));
	// ReLu
	//return ((x <= 0) ? 0 : x);
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
	// ReLu Prime
	//return ((x <= 0) ? 0 : 1);
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
