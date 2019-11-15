#include "readMNIST.h"
#include "perceptronSimple.h"

int main(){

    // Dataset filenames
    string X_train_file = "mnist/train-images-idx3-ubyte";
    string y_train_file = "mnist/train-labels-idx1-ubyte";
    string X_test_file = "mnist/t10k-images-idx3-ubyte";
    string y_test_file = "mnist/t10k-labels-idx1-ubyte";

    // Data info
    int number_of_train_images = 60000;
    int number_of_test_images = 10000;
    int image_size = 28 * 28;

    struct timeval start, end;

    vector<vector<double> > X_train, X_test;
    vector<int> y_train(number_of_train_images, 0.0);
    vector<int> y_test(number_of_test_images, 0.0);

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

    MnistSimplePerceptron perceptron = MnistSimplePerceptron(image_size);

    cout << "Training..." << endl;


    gettimeofday(&start, NULL);
    perceptron.train(X_train, y_train);
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

/*
    //read MNIST iamge into Armadillo mat vector
    vector<arma::mat> vec;
    read_Mnist(filename, vec);
    cout<<vec.size()<<endl;
    cout<<vec[0].size()<<endl;
*/
/*
    //read MNIST iamge into OpenCV Mat vector
    vector<cv::Mat> vec;
    read_Mnist(filename, vec);
    cout<<vec.size()<<endl;
    imshow("1st", vec[8]);
    waitKey();
*/
/*
    //read MNIST iamge into double vector
    vector<vector<double> > vec;
    read_Mnist(filename, vec);
    cout<<vec.size()<<endl;
    cout<<vec[0].size()<<endl;
*/

//    string filename = "mnist/t10k-labels-idx1-ubyte";
//    int number_of_images = 10000;

/*
    //read MNIST label into double vector
    vector<double> vec(number_of_images);
    read_Mnist_Label(filename, vec);
    cout<<vec.size()<<endl;
*/

/*
    //read MNIST label into armadillo colvec
    //if you want rowvec, just use .t()
    arma::colvec vec = arma::zeros<arma::colvec>(number_of_images);
    read_Mnist_Label(filename, vec);
    cout<<vec.size()<<endl;
*/
    return 0;
}
