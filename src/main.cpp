#include <stdlib.h>
#include "readMNIST.h"
#include "perceptronSimple.h"
#include "perceptronMultiLayer.h"

int main(int argc, char** argv){

    // Dataset filenames
    string X_train_file = "mnist/train-images-idx3-ubyte";
    string y_train_file = "mnist/train-labels-idx1-ubyte";
    string X_test_file = "mnist/t10k-images-idx3-ubyte";
    string y_test_file = "mnist/t10k-labels-idx1-ubyte";

    // Data info
    int number_of_train_images = 60000;
    int number_of_test_images = 10000;
    int image_size = 28 * 28;

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

    //MnistSimplePerceptron perceptron = MnistSimplePerceptron(image_size);

    int hidden_layers = atoi(argv[1]);
    vector<int> sizes(2+hidden_layers);

    sizes[0] = image_size;
    sizes[hidden_layers+1] = 10;

    for(int i = 0; i < hidden_layers; i++){
        sizes[i+1] = atoi(argv[i+2]);
    }

    int epochs = atoi(argv[hidden_layers+2]);
    int mini_batch_size = atoi(argv[hidden_layers+3]);
    int eta = atof(argv[hidden_layers+4]);

    Network perceptron = Network(sizes);

    cout << "Training..." << endl;


    gettimeofday(&start, NULL);
    perceptron.train(X_train, y_train, X_test, y_test, epochs, mini_batch_size, eta);
    gettimeofday(&end, NULL);

    long seconds = (end.tv_sec - start.tv_sec);
	long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    cout << "Time taken by program is : " << micros/1000000.0 << " sec." << endl;

    int success_train = 0;

    for(int i = 0; i < number_of_train_images; i++){
        int prediction = perceptron.predict(X_train[i]);

        if(prediction == y_train[i]){
            success_train++;
        }
    }

    cout << "Train success: " << success_train << "/" << number_of_train_images;
    cout << ", " << 100.0*success_train/number_of_train_images << "%" << endl;


    int success_test = 0;

    for(int i = 0; i < number_of_test_images; i++){
        int prediction = perceptron.predict(X_test[i]);

        if(prediction == y_test[i]){
            success_test++;
        }
    }

    cout << "Test success: " << success_test << "/" << number_of_test_images;
    cout << ", " << 100.0*success_test/number_of_test_images << "%" << endl;

    return 0;
}
