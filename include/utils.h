#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <assert.h>

using namespace std;

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    unsigned int a_size = a.size();
    assert(a_size == b.size());
    std::vector<T> result(a_size);

	for(unsigned int i = 0; i < a_size; i++){
		result[i] = a[i]+b[i];
	}

    return result;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
    unsigned int a_size = a.size();
    assert(a_size == b.size());
    std::vector<T> result(a_size);

	for(unsigned int i = 0; i < a_size; i++){
		result[i] = a[i]-b[i];
	}

    return result;
}

template <typename T>
std::vector<T> operator*(const std::vector<T>& a, const std::vector<T>& b)
{
    unsigned int a_size = a.size();
    assert(a_size == b.size());
    std::vector<T> result(a_size);

	for(unsigned int i = 0; i < a_size; i++){
		result[i] = a[i]*b[i];
	}

    return result;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& a)
{
    unsigned int a_size = a.size();
    std::vector<T> result(a_size);

	for(unsigned int i = 0; i < a_size; i++){
		result[i] = -a[i];
	}

    return result;
}

template <typename C, typename T>
std::vector <T> operator* (const C &c, const std::vector <T> &a)
{
	unsigned int a_size = a.size();
    std::vector<T> result(a_size);

	for(unsigned int i = 0; i < a_size; i++){
        result[i] = c*a[i];
	}

    return result;
}

template <typename T>
std::vector<std::vector<T> > matrix_multiplication(const std::vector<std::vector<T> > &a, const std::vector<std::vector<T> > &b){
    unsigned int a_rows = a.size();
    unsigned int b_cols = b[0].size();
    unsigned int a_cols = a[0].size();

    assert(a_cols == b.size());

    std::vector<std::vector<T> > result(a_rows, std::vector<T>(b_cols, 0.0));

    for(int i = 0; i < a_rows; i++){
        for(int j = 0; j < b_cols; j++){
            for(int k = 0; k < a_cols; k++){
                result[i][j] += a[i][k]*b[k][j];
            }
        }
    }

    return result;
}

template <typename T>
std::vector<T> matrix_multiplication(const std::vector<std::vector<T> > &a, const std::vector<T> &b){
    unsigned int a_rows = a.size();
    unsigned int a_cols = a[0].size();

    assert(a_cols == b.size());

    std::vector<T> result(a_rows, 0.0);

    for(int i = 0; i < a_rows; i++){
        for(int k = 0; k < a_cols; k++){
            result[i] += a[i][k]*b[k];
        }
    }

    return result;
}

template<typename T>
std::vector<std::vector<T> > transpose(const std::vector<std::vector<T> > &a){

    unsigned int a_rows = a.size();
    unsigned int a_cols = a[0].size();

    std::vector<std::vector<T> > result(a_cols, std::vector<T>(a_rows));

    for(int i = 0; i < a_rows; i++){
        for(int j = 0; j < a_cols; j++){
            result[j][i] = a[i][j];
        }
    }

    return result;
}
