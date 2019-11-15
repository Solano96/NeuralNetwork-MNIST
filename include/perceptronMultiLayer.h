
#include <vector>

class Network(){
private:
	double num_layers;
	vector<double> sizes;
	vector<vector<double> > biases;
	vector<vector<vector<double> > > weights;

public:
	Network(vector<double> sizes);

	vector<double> feedForward(vector<double> a);

	void SGD(vector<vector<double> >, int epochs, int mini_batch_size, double eta);


};
