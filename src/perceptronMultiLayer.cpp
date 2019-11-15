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

vector<double> Network::feedForward(vector<double> a);

void Network::SGD(vector<vector<double> >, int epochs, int mini_batch_size, double eta);




for(int i = 0; i < input_size; i++){
	weights.push_back(distribution(generator)*10);
}
