#include "perceptronMultiLayer.h"

Network::Network(vector<int> sizes_){
	this->num_layers = sizes_.size();
	this->sizes = sizes_;

	biases_fill(this->biases, this->num_layers, this->sizes);

	default_random_engine generator;
	normal_distribution<double> distribution(0.0,1.0);

	weights_fill(this->weights, this->num_layers, this->sizes, true);
}

void Network::biases_fill(vector<vector<double> > &b, int n_layers, vector<int> &l_sizes){
	b.clear()

	for(int i = 1; i < n_layers; i++){
		vector<double> b_layer;
		for(int j = 0; j < l_sizes[i]; j++){
			b_layer.push_back(0.0);
		}
		b.push_back(bias_layer);
	}
}

void Network::weights_fill(vector<vector<vector<double> > > &w, int n_layers, vector<int> &l_sizes, bool random_w){
	w.clear();

	default_random_engine generator;
	normal_distribution<double> distribution(0.0,1.0);

	for(int i = 1; i < n_layers; i++){
		vector<double> layers_w;
		for(int j = 0; j < l_sizes[i]; j++){
			vector<vector<double> > node_w;
			for(int k = 0; k < l_size[i-1]; k++){
				if(random_w){
					node_w.push_back(distribution(generator)*10);
				}
				else{
					node_w.push_back(0.0);
				}
			}
			layers_w.push_back(node_w)
		}
		w.push_back(layers_w);
	}

}

vector<double> Network::feedForward(vector<double> input){
	vector<double> a = input;

	for(int i = 0; i < this->num_layers-1; i++){
		vector<double> layer_outputs;
		for(int j = 0; j < this->sizes[i]; j++){
			double dot_product = inner_product(a.begin(), a.end(), this->weights[i][j].begin(), 0);
			double node_output = sigmoid(dot_product + this->biases[i][j]);
			layers_outputs.push_back(node_output);
		}
		a = layer_outputs;
	}

	return a;
}

void Network::SGD(vector<vector<double> > &x_train, vector<int> &y_train, int epochs, int mini_batch_size, double eta){

	vector<int> index;
	int train_size = y_train.size();

	for(int i = 0; i < train_size; i++){
		index.push_back(i)
	}

	for(int i = 0; i < epochs; i++){
		random_shuffle(index.begin(), index.end());

		vector<vector<double> > x_mini_batch;
		vector<int> y_mini_batch;

		for(int j = 0; j < mini_batch_size; j++){
			x_mini_batch.push_back(x_train[index[j]]);
			y_mini_batch.push_back(y_train[index[j]]);
		}

		update_mini_batch(x_mini_batch, y_mini_batch, eta);

		cout << "Epoch " << i << "/" << epochs << endl;
	}
}

void Network::update_mini_batch(vector<vector<double> > &x_mini_batch, vector<int> &y_mini_batch, double eta){

	vector<vector<double> > nabla_b;
	vector<vector<vector<double> > > nabla_w;

	biases_fill(nabla_b, num_layers, sizes);
	weights_fill(nabla_w, num_layers, sizes, false);

	for(int i = 0; i < x_mini_batch.size(); i++){

	}
}


void Network::backprop(vector<double> &x, int &y){
	biases_fill(nabla_b, num_layers, sizes);
	weights_fill(nabla_w, num_layers, sizes, false);

	vector<double> activation = x;
	vector<vector<double> >  activations;

	activations.push_back(activation);

	vector<vector<double> > zs;

	for(int i = 0; i < this->num_layers; i++){
		vector<double> z;
		vector<double> z_sigmoid;

		for(int j = 0; j < this->sizes[i]; j++){
			double dot_product = inner_product(activation.begin(), activation.end(), this->weights[i][j].begin(), 0);
			double out = dot_product + this->biases[i][j];
			z.push_back(out);
			z_sigmoid.push_back(sigmoid(out));
		}

		zs.push_back(z);
		activation = z_sigmoid;
		activations.push_back(activation);
	}

}
