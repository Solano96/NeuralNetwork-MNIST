#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <assert.h>

using namespace std;

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    int a_size = a.size();
    assert(a_size == b.size());
    std::vector<T> result(a_size);

	for(int i = 0; i < a_size; i++){
		result[i] = a[i]+b[i];
	}

    return result;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
    int a_size = a.size();
    assert(a_size == b.size());
    std::vector<T> result(a_size);

	for(int i = 0; i < a_size; i++){
		result[i] = a[i]-b[i];
	}

    return result;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& a)
{
    int a_size = a.size();
    std::vector<T> result(a_size);

	for(int i = 0; i < a_size; i++){
		result[i] = -a[i];
	}

    return result;
}

template <typename C, typename T>
std::vector <T> operator* (const C &c, const std::vector <T> &a)
{
	int a_size = a.size();
    std::vector<T> result(a_size);

	for(int i = 0; i < a_size; i++){
        result[i] = c*a[i];
	}

    return result;
}
