#include "utils.h"

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}

template <typename T>
std::vector<std::vector<T> > operator+(const std::vector<std::vector<T> >& a, const std::vector<std::vector<T> >& b)
{
    assert(a.size() == b.size());

    std::vector<std::vector<T> > result;

	for(int i = 0; i < a.size(); i++){
		result.push_back(a[i]+b[i]);
	}

    return result;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::minus<T>());
    return result;
}

template <typename T>
std::vector<std::vector<T> > operator-(const std::vector<std::vector<T> >& a, const std::vector<std::vector<T> >& b)
{
    assert(a.size() == b.size());

    std::vector<std::vector<T> > result;
	int a_size = a.size();

	for(int i = 0; i < a_size; i++){
		result.push_back(a[i]-b[i]);
	}

    return result;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& a)
{
    std::vector<T> result;
	int a_size = a.size();

	for(int i = 0; i < a_size; i++){
		result.push_back(-a[i]);
	}

    return result;
}

template <typename T>
std::vector<std::vector<T> > operator-(const std::vector<std::vector<T> >& a)
{
    std::vector<std::vector<T> > result;
	int a_size = a.size();

	for(int i = 0; i < a_size; i++){
		result.push_back(-a[i]);
	}

    return result;
}

template <typename T>
std::vector<T> operator*(const T& c, const std::vector<T>& a)
{
    std::vector<T> result;
	int a_size = a.size();

	for(int i = 0; i < a_size; i++){
		result.push_back(c*a[i]);
	}

    return result;
}

template <typename T>
std::vector<std::vector<T> > operator*(const T& c, const std::vector<std::vector<T> >& a)
{
    std::vector<std::vector<T> > result;
	int a_size = a.size();

	for(int i = 0; i < a_size; i++){
		result.push_back(c*a[i]);
	}

    return result;
}
