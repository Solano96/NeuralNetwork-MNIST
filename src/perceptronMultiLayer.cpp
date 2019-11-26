#include "perceptronMultiLayer.h"
#include "utils.h"

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

	// Se va a suponer en principio que mini_batch_size es divisor de train_size
	// Posiblemente se modifique mas adelante

	vector<int> index;
	int train_size = y_train.size();

	for(int i = 0; i < train_size; i++){
		index.push_back(i)
	}

	for(int i = 0; i < epochs; i++){
		random_shuffle(index.begin(), index.end());

		vector<vector<vector<double> > > x_mini_batches;
		vector<vector<int> > y_mini_batches;

		for(int j = 0; j < train_size; j += mini_batch_size){
			vector<vector<double> > x_mini_batch;
			vector<int> y_mini_batch;

			for(int k = j; k < j+mini_batch_size; k++){
				x_mini_batch.push_back(x_train[index[k]]);
				y_mini_batch.push_back(y_train[index[k]]);
			}

			x_mini_batches.push_back(x_mini_batch);
			y_mini_batches.push_back(y_mini_batch);
		}

		int num_mini_batches = y_mini_batches.size();

		for(int j = 0; j < num_mini_batches; j++){
			update_mini_batch(x_mini_batch, y_mini_batch, eta);
		}

		cout << "Epoch " << i << "/" << epochs << endl;
	}
}

// SIN ACABAR
void Network::update_mini_batch(vector<vector<double> > &x_mini_batch, vector<int> &y_mini_batch, double eta){
	vector<vector<double> > nabla_b;
	vector<vector<vector<double> > > nabla_w;

	biases_fill(nabla_b, this->num_layers, this->sizes);
	weights_fill(nabla_w, this->num_layers, this->sizes, false);

	int mini_batch_size = x_mini_batch.size();

	for(int i = 0; i < mini_batch_size; i++){
		vector<vector<double> > delta_nabla_b;
		vector<vector<vector<double> > > delta_nabla_w;
		backprop(x_mini_batch[i], y_mini_batch[i], delta_nabla_b, delta_nabla_w);

		nabla_b = nabla_b + delta_nabla_b;

		for(int j = 0; j < this->num_layers; j++){
			nabla_w[j] = nabla_w[j] + delta_nabla_w[j];
		}

	}
}


void Network::backprop(vector<double> &x, int &y, vector<vector<double> > &nabla_b, vector<vector<vector<double> > > &nabla_w){
	biases_fill(nabla_b, this->num_layers, this->sizes);
	weights_fill(nabla_w, this->num_layers, this->sizes, false);

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

	int activations_size = activations.size();

	vector<double> partial_derivatives = cost_derivative(activations[activations_size-1], y);
	vector<double> sigmoid_prime_vector = sigmoid_prime(zs[zs.size()-1]);

	int delta_size = sigmoid_prime_vector.size();
	vector<double> delta;

	for(int i = 0; i < delta_size; i++){
		delta.push_back(partial_derivatives[i]*sigmoid_prime_vector[i]);
	}

	nabla_b[num_layers-1] = delta;

	for(int i = 0; i < this->sizes[num_layers-1]; i++){
		for(int j = 0; j < this->sizes[num_layers-2]; j++){
			nabla_w[num_layers-1][i][j] = delta[i]*activations[activations.size()-2][j];
		}
	}

	for(int i = 2; i < num_layers; i++){
		vector<double> z = zs[num_layers-i];
		vector<double> sp = sigmoid_prime(z);

		vector<double> temp_delta;

		for(int j = 0; j < this->sizes[num_layers-i]; j++){
			double dot_product = 0;
			for(int k = 0; k < this->sizes[num_layers-i+1]; k++){
				dot_product += delta[k]*this->weights[num_layers-i+1][k][j];
			}
			temp_delta.push_back(dot_product*sp[j]);
		}

		delta = temp_delta;
		nabla_b[num_layers-i] = delta;

		for(int j = 0; j < this->sizes[num_layers-i]; j++){
			for(int k = 0; k < this->sizes[num_layers-i-1]; k++){
				nabla_w[num_layers-i][j][k] = delta[j]*activations[num_layers-i-1][k];
			}
		}
	}

}

void Network::evaluate(){}


vector<double> Network::cost_derivative(vector<double> outputs_activations, vector<int> y){
	vector<double> partial_derivatives;
	int y_size = y.size();

	for(int i = 0; i < y_size(); y++){
		partial_derivatives.push_back(outputs_activations[i]-y[i]);
	}
	return partial_derivatives;
}
