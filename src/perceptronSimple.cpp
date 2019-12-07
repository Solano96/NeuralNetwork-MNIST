#include "perceptronSimple.h"


SimplePerceptron::SimplePerceptron(int input_size_){
    input_size = input_size_;
    bias = 0;

    default_random_engine generator;
    normal_distribution<double> distribution(0.0,1.0);

    for(int i = 0; i < input_size; i++){
        weights.push_back(distribution(generator)*10);
    }
}

void SimplePerceptron::train(vector<vector<double> > &dataset, vector<int> &labels, int epochs, double learning_rate){
    bool convergence = false;
    int dataset_size = dataset.size();
    vector<int> index;

    for(int i = 0; i < dataset_size; i++){
        index.push_back(i);
    }

    for(int epoch = 0; epoch < epochs && !convergence; epoch++){
        convergence = true;
        random_shuffle(index.begin(), index.end());

        for(int i = 0; i < dataset_size; i++){
            vector<double> data = dataset[index[i]];
            int label = labels[index[i]];
            int output = ((get_output(data) < 0.0) ? 0 : 1);

            if(output != label){
                int dir = label-output;
                convergence = false;

                for(int j = 0; j < input_size; j++){
                    weights[j] += learning_rate*dir*data[j];
                }

                bias += learning_rate*dir;
            }
        }
    }
}

double SimplePerceptron::get_output(vector<double> input){
    return inner_product(weights.begin(), weights.end(), input.begin(), 0.0)+bias;
}



MnistSimplePerceptron::MnistSimplePerceptron(int input_size_){
    input_size = input_size_;

    for(int i = 0; i < 10; i++){
        neurons.push_back(SimplePerceptron(input_size));
    }
}

void MnistSimplePerceptron::train(vector<vector<double> > &dataset, vector<int> &label, int epochs, double learning_rate){
    int dataset_size = dataset.size();

    #pragma omp parallel for
    for(int i = 0; i < 10; i++){
        vector<int> label_i;

        for(int j = 0; j < dataset_size; j++){
            label_i.push_back( (label[j] != i) ? 0 : 1);
        }

        neurons[i].train(dataset, label_i, epochs, learning_rate);
    }
}

int MnistSimplePerceptron::predict(vector<double> &data){
    vector<double> outputs;
    int max = 0;

    for(int i = 0; i < 10; i++)
        outputs.push_back(neurons[i].get_output(data));

    for(int i = 1; i < 10; i++)
        max = ((outputs[max] < outputs[i]) ? i : max);

    return max;
}

double MnistSimplePerceptron::get_accuracy(vector<vector<double> > &x, vector<int> &y){
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
