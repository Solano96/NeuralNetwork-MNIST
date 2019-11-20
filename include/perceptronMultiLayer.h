#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

class Network(){
private:
	double num_layers;
	vector<int> sizes;
	vector<vector<double> > biases;
	vector<vector<vector<double> > > weights;

	biases_fill(vector<vector<double> > &b, int n_layers, vector<int> &l_sizes);
	weights_fill(vector<vector<vector<double> > > &w, int n_layers, vector<int> &l_sizes, bool random_w);

public:
	Network(vector<int> sizes);

	vector<double> feedForward(vector<double> a);

	void SGD(vector<vector<double> > &x_train, vector<int> &y_train, int epochs, int mini_batch_size, double eta);

	void update_mini_batch(vector<vector<double> > &x_mini_batch, vector<int> &y_mini_batch, double eta);

	void backprop()

};
