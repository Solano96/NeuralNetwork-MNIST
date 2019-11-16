#include "perceptronMultiLayer.h"

Network::Network(vector<double> sizes_){
	num_layers = sizes_.size();
	sizes = sizes_;

	// Initialize biases
	for(int i = 1; i < num_layers; i++){
		vector<double> bias_layer;
		for(int j = 0; j < sizes[i]; j++){
			bias_layer.push_back(0.0);
		}
		biases.push_back(bias_layer);
	}

	default_random_engine generator;
	normal_distribution<double> distribution(0.0,1.0);

	// Initialize weights
	for(int i = 1; i < num_layers; i++){
		vector<double> layers_weights;
		for(int j = 0; j < sizes[i]; j++){
			vector<vector<double> > node_weights;
			for(int k = 0; k < size[i-1]; k++){
				node_weights.push_back(distribution(generator)*10);
			}
			layers_weights.push_back(node_weights)
		}
		weights.push_back(layers_weights);
	}
}

vector<double> Network::feedForward(vector<double> input){
	vector<double> a = input;

	for(int i = 0; i < num_layers-1; i++){
		vector<double> layer_outputs;
		for(int j = 0; j < sizes[i]; j++){
			double dot_product = inner_product(a.begin(), a.end(), weights[i][j].begin(), 0);
			double node_output = sigmoid(dot_product + biases[i][j]);
			layers_outputs.push_back(node_output);
		}
		a = layer_outputs;
	}

	return a;
}

void Network::SGD(vector<vector<double> > x_train, vector<int> y_train, int epochs, int mini_batch_size, double eta){

	vector<int> index;
	int train_size = y_train.size();

	for(int i = 0; i < train_size; i++){
		index.push_back(i)
	}

	for(int i = 0; i < epochs; i++){
		random_shuffle(index.begin(), index.end());

		vector<vector<double> > x_mini_batch;
		vector<vector<double> > y_mini_batch;

		for(int j = 0; j < mini_batch_size; j++){
			x_mini_batch.push_back(x_train[index[j]]);
			y_mini_batch.push_back(y_train[index[j]]);
		}


	}
}
