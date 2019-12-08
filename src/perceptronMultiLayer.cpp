#include "perceptronMultiLayer.h"
#include "matrixOperation.h"

MLP::MLP(vector<int> sizes_){
	this->num_layers = sizes_.size();
	this->sizes = sizes_;

	biases_fill(this->biases, this->num_layers, this->sizes);

	default_random_engine generator;
	normal_distribution<double> distribution(0.0,1.0);

	weights_fill(this->weights, this->num_layers, this->sizes, true);
}

void MLP::biases_fill(vector<vector<double> > &b, int n_layers, vector<int> &l_sizes){
	b.clear();

	for(int i = 1; i < n_layers; i++){
		b.push_back(vector<double>(l_sizes[i], 0.0));
	}
}

void MLP::weights_fill(vector<vector<vector<double> > > &w, int n_layers, vector<int> &l_sizes, bool random_w){
	w.clear();
	default_random_engine generator;
	normal_distribution<double> distribution(0.0,1.0);

	for(int i = 1; i < n_layers; i++){
		vector<vector<double> > layers_w(l_sizes[i], vector<double>(l_sizes[i-1]));

		for(int j = 0; j < l_sizes[i]; j++){
            if(random_w){
    			for(int k = 0; k < l_sizes[i-1]; k++){
					layers_w[j][k] = distribution(generator)/sqrt(l_sizes[i-1]);
    			}
            }
            else{
                layers_w[j] = vector<double>(l_sizes[i-1], 0.0);
            }
		}
		w.push_back(layers_w);
	}
}

vector<double> MLP::feedForward(vector<double> &input){
	vector<double> a = input;

    for(int i = 1; i < this->num_layers; i++){
    	vector<double> z = matrix_multiplication(this->weights[i-1], a) + this->biases[i-1];
		a = (i == this->num_layers-1) ? sigmoid(z) : sigmoid(z);
    }

	return a;
}

double MLP::loss_function(vector<vector<double> > &x, vector<vector<int> > &y){
		double loss = 0.0;
		int x_size = x.size();

		for(int i = 0; i < x_size; i++){
			vector<double> y_pred = feedForward(x[i]);

			for(int j = 0; j < 10; j++){
				loss += y[i][j]*log(y_pred[j]);
			}
		}

		return loss/-x_size;
}

void MLP::SGD(vector<vector<double> > &x_train, vector<vector<int> > &y_train, vector<vector<double> > &x_test,
	vector<vector<int> > &y_test, int epochs, int mini_batch_size, double eta){

	string filename = string("resultados/mpl_");

	for(int i = 1; i < this->num_layers-1; i++){
		filename += to_string(this->sizes[i]) + string("_");
	}

	filename += to_string(epochs) + string("_") + to_string(mini_batch_size) + string(".txt");

	ofstream outfile (filename);

	vector<int> index;
	int train_size = y_train.size();

	for(int i = 0; i < train_size; i++){
		index.push_back(i);
	}

	for(int i = 0; i < epochs; i++){
		random_shuffle(index.begin(), index.end());

		vector<vector<vector<double> > > x_mini_batches;
		vector<vector<vector<int> > > y_mini_batches;

		for(int j = 0; j < train_size; j += mini_batch_size){
			vector<vector<double> > x_mini_batch;
			vector<vector<int> > y_mini_batch;

			for(int k = j; k < j+mini_batch_size; k++){
				x_mini_batch.push_back(x_train[index[k]]);
				y_mini_batch.push_back(y_train[index[k]]);
			}

			x_mini_batches.push_back(x_mini_batch);
			y_mini_batches.push_back(y_mini_batch);
		}

		int num_mini_batches = y_mini_batches.size();

		struct timeval start, end;

		gettimeofday(&start, NULL);

		//#pragma omp parallel for
		for(int j = 0; j < num_mini_batches; j++){
			update_mini_batch(x_mini_batches[j], y_mini_batches[j], eta);
		}

		gettimeofday(&end, NULL);

		long seconds = (end.tv_sec - start.tv_sec);
		long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

		outfile << "Epoch " << i+1 << "/" << epochs << endl;
		cout << "Epoch " << i+1 << "/" << epochs << endl;

		double train_loss = loss_function(x_train, y_train);
		double test_loss = loss_function(x_test, y_test);

		double train_accuracy = get_accuracy(x_train, y_train);
		double test_accuracy = get_accuracy(x_test, y_test);

		outfile << " - " << micros/1000000 << "s";
		outfile << " - loss: " << train_loss << " - acc: " << train_accuracy;
		outfile << " - val_loss: " << test_loss << " - val_acc: " << test_accuracy << endl;

		cout << " - " << micros/1000000 << "s";
		cout << " - loss: " << train_loss << " - acc: " << train_accuracy;
		cout << " - val_loss: " << test_loss << " - val_acc: " << test_accuracy << endl;
	}

	outfile << endl << endl;

	for(int i = 0; i < x_test.size(); i++){
		outfile << predict(x_test[i]);
	}

	outfile << endl << endl;

	outfile.close();
}


void MLP::update_mini_batch(vector<vector<double> > &x_mini_batch, vector<vector<int> > &y_mini_batch, double eta){
	vector<vector<double> > nabla_b;
	vector<vector<vector<double> > > nabla_w;

	biases_fill(nabla_b, this->num_layers, this->sizes);
	weights_fill(nabla_w, this->num_layers, this->sizes, false);

	int mini_batch_size = x_mini_batch.size();

	//#pragma omp parallel reduction(+ : nabla_b, + : nabla_w[ : this->num_layers])
	for(int i = 0; i < mini_batch_size; i++){
		vector<vector<double> > delta_nabla_b;
		vector<vector<vector<double> > > delta_nabla_w;

		backprop(x_mini_batch[i], y_mini_batch[i], delta_nabla_b, delta_nabla_w);

		nabla_b = nabla_b + delta_nabla_b;

		for(int j = 0; j < this->num_layers-1; j++){
			nabla_w[j] = nabla_w[j] + delta_nabla_w[j];
		}
	}

	double c = 1.0*eta/mini_batch_size;

	this->biases = this->biases - c*nabla_b;

	for(int j = 0; j < this->num_layers-1; j++){
		this->weights[j] = this->weights[j] - c*nabla_w[j];
	}

}

void MLP::backprop(vector<double> &x, vector<int> &y, vector<vector<double> > &nabla_b, vector<vector<vector<double> > > &nabla_w){
	// Inicializamos nabla_w y nabla_b
	biases_fill(nabla_b, this->num_layers, this->sizes);
	weights_fill(nabla_w, this->num_layers, this->sizes, false);

	// Inicilizamos la activacion al vector de entrada
	vector<double> activation = x;
	vector<vector<double> >  activations;

	activations.push_back(activation);

	vector<vector<double> > zs;

	for(int i = 1; i < this->num_layers; i++){
		vector<double> z = matrix_multiplication(this->weights[i-1], activation) + this->biases[i-1];
		zs.push_back(z);
		activation = (i == this->num_layers-1) ? sigmoid(z) : sigmoid(z);
		activations.push_back(activation);
	}

	int activations_size = activations.size();
    int zs_size = zs.size();

	// Obtener derivadas parciales
	vector<double> delta = cost_derivative(activations[activations_size-1], y);

	nabla_b[this->num_layers-2] = delta;


	for(int i = 0; i < this->sizes[this->num_layers-1]; i++){
		nabla_w[this->num_layers-2][i] = delta[i]*activations[activations_size-2];
	}

	for(int i = 2; i < num_layers; i++){
		vector<double> dA = matrix_multiplication(transpose(this->weights[this->num_layers-i]), delta);
		vector<double> z = zs[zs_size-i];
		vector<double> sp = sigmoid_prime(z);

		delta = dA * sp;

		nabla_b[this->num_layers-i-1] = delta;

		for(int j = 0; j < this->sizes[num_layers-i]; j++){
			nabla_w[this->num_layers-i-1][j] = delta[j]*activations[activations_size-i-1];
		}
	}
}


vector<double> MLP::cost_derivative(vector<double> &outputs_activations, vector<int> &y){
	vector<double> partial_derivatives(10);

	for(int i = 0; i < 10; i++){
		partial_derivatives[i] = outputs_activations[i]-y[i];
	}

	return partial_derivatives;
}

void MLP::train(vector<vector<double> > &dataset, vector<int> &label, vector<vector<double> > &x_test, vector<int> &y_test,
	int epochs, int mini_batch_size, double eta){

    int label_size = label.size();

    vector<vector<int> > y_train_categorical(label_size);

    for(int i = 0; i < label_size; i++){
        for(int j = 0; j < 10; j++){
            y_train_categorical[i].push_back((label[i] == j) ? 1 : 0);
        }
    }

    int y_test_size = y_test.size();

    vector<vector<int> > y_test_categorical(y_test_size);

    for(int i = 0; i < y_test_size; i++){
        for(int j = 0; j < 10; j++){
            y_test_categorical[i].push_back((y_test[i] == j) ? 1 : 0);
        }
    }

	SGD(dataset, y_train_categorical, x_test, y_test_categorical, epochs, mini_batch_size, eta);
}


int MLP::predict(vector<double> &data){
    vector<double> outputs = feedForward(data);
    int max = 0;

    for(int i = 1; i < 10; i++)
        max = ((outputs[max] < outputs[i]) ? i : max);

    return max;
}

double MLP::get_accuracy(vector<vector<double> > &x, vector<vector<int> > &y){
	int num_success = 0;
	int x_size = x.size();

	for(int i = 0; i < x_size; i++){
		int prediction = predict(x[i]);
		int y_value = 0;

		for(int j = 1; j < 10; j++){
			if(y[i][j] == 1){
				y_value = j;
			}
		}

		if(prediction == y_value){
			num_success++;
		}
	}

	return 1.0*num_success/x_size;
}

double MLP::get_accuracy(vector<vector<double> > &x, vector<int> &y){
	int num_success = 0;
	int x_size = x.size();

	for(int i = 0; i < x_size; i++){
		int prediction = predict(x[i]);

		if(prediction == y[i]){
			num_success++;
		}
	}

	return 1.0*num_success/x_size;
}
