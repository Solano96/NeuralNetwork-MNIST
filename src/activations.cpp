
#include "activations.h"
#include <math.h>       /* exp */


double sigmoid(double x){
	return 1.0/(1.0+exp(-x));
}

double sigmoid_prime(double x){
	return sigmoid(x)*(1-sigmoid(x));
}
