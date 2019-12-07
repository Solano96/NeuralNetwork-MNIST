#include <stdlib.h>
#include <string>
#include "readMNIST.h"
#include "perceptronSimple.h"
#include "perceptronMultiLayer.h"
#include "convolutionalNeuralNetwork.h"

int main(int argc, char** argv){

    // Dataset filenames
    string X_train_file = "mnist/train-images-idx3-ubyte";
    string y_train_file = "mnist/train-labels-idx1-ubyte";
    string X_test_file = "mnist/t10k-images-idx3-ubyte";
    string y_test_file = "mnist/t10k-labels-idx1-ubyte";

    // Data info
    int number_of_train_images = 60000;
    int number_of_test_images = 10000;

    vector<vector<double> > X_train, X_test;
    vector<int> y_train(number_of_train_images, 0.0);
    vector<int> y_test(number_of_test_images, 0.0);

    struct timeval start, end;

    cout << "Reading data..." << endl;

    // Dataset training
    read_Mnist(X_train_file, X_train);
    read_Mnist_Label(y_train_file, y_train);

    // Dataset test
    read_Mnist(X_test_file, X_test);
    read_Mnist_Label(y_test_file, y_test);

    // Normalize dataset
    normalize_dataset(X_train);
    normalize_dataset(X_test);

    /****************************** SIMPLE ******************************/

    if(string(argv[1]) == "simple"){
        int epochs = atoi(argv[2]);
        double eta = atof(argv[3]);

        MnistSimplePerceptron perceptron(X_train[0].size());

        cout << "Training..." << endl;

        gettimeofday(&start, NULL);
        perceptron.train(X_train, y_train);
        gettimeofday(&end, NULL);

        cout << "Train accuracy: " << perceptron.get_accuracy(X_train, y_train) << endl;
        cout << "Test accuracy: " << perceptron.get_accuracy(X_test, y_test) << endl;
    }

    /****************************** MULTILAYER ******************************/

    else if(string(argv[1]) == "multicapa"){
        int hidden_layers = atoi(argv[2]);
        vector<int> sizes(2+hidden_layers);

        sizes[0] = X_train[0].size();
        sizes[hidden_layers+1] = 10;

        for(int i = 0; i < hidden_layers; i++){
            sizes[i+1] = atoi(argv[i+3]);
        }

        int epochs = atoi(argv[hidden_layers+3]);
        int mini_batch_size = atoi(argv[hidden_layers+4]);
        double eta = atof(argv[hidden_layers+5]);

        MLP multiLayerPerceptron(sizes);

        cout << "Training..." << endl;

        gettimeofday(&start, NULL);
        multiLayerPerceptron.train(X_train, y_train, X_test, y_test, epochs, mini_batch_size, eta);
        gettimeofday(&end, NULL);
    }

    /****************************** CONVOLUTIONAL ******************************/

    else if(string(argv[1]) == "convolucional"){
            int hidden_layers = atoi(argv[2]);
            vector<int> sizes(2+hidden_layers);

            sizes[0] = X_train[0].size();
            sizes[hidden_layers+1] = 10;

            for(int i = 0; i < hidden_layers; i++){
                sizes[i+1] = atoi(argv[i+3]);
            }

            int epochs = atoi(argv[hidden_layers+3]);
            int mini_batch_size = atoi(argv[hidden_layers+4]);
            double eta = atof(argv[hidden_layers+5]);

            CNN convolutionalNeuralNetwork(sizes);

            cout << "Training..." << endl;

            gettimeofday(&start, NULL);
            convolutionalNeuralNetwork.train(X_train, y_train, X_test, y_test, epochs, mini_batch_size, eta);
            gettimeofday(&end, NULL);
    }
    else{
        cout << "ERROR: parámetros incorrectos." << endl;
        cout << "./main <nombre-red-neuronal> <n-capas-ocultas> <nodos-capa-1> ... <nodos-capa-n> <epocas> <tam-mini-batch> <eta>" << endl;
        cout << "<nombre-red-neuronal>: simple | multicapa | convolucional" << endl;
        cout << "En el caso de la simple solo se necesitan las épocas y eta." << endl;
    }

    long seconds = (end.tv_sec - start.tv_sec);
	long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    cout << "Time taken by program is : " << micros/1000000.0 << " sec." << endl;
    /*cout << "Train accuracy: " << 100.0*perceptron.get_accuracy(X_train, y_train) << "%" << endl;
    cout << "Test accuracy: " << 100.0*perceptron.get_accuracy(X_test, y_test) << "%" << endl;*/

    return 0;
}
