#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

class Network(){
private:
	double num_layers;
	vector<double> sizes;
	vector<vector<double> > biases;
	vector<vector<vector<double> > > weights;

public:
	Network(vector<double> sizes);

	vector<double> feedForward(vector<double> a);

	void SGD(vector<vector<double> > x_train, vector<int> y_train, int epochs, int mini_batch_size, double eta);

	void update_mini_batch(vector<vector<double> > mini_batch, double eta);


};
