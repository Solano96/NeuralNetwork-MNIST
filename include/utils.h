#include <vector>
#include <algorithm>
#include <functional>
#include <assert.h>

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b);

template <typename T>
std::vector<std::vector<T> > operator+(const std::vector<std::vector<T> >& a, const std::vector<std::vector<T> >& b);

template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b);

template <typename T>
std::vector<T> operator-(const std::vector<T>& a);

template <typename T>
std::vector<std::vector<T> > operator-(const std::vector<std::vector<T> >& a);

template <typename T>
std::vector<T> operator*(const T& c, const std::vector<T>& a);

template <typename T>
std::vector<std::vector<T> > operator*(const T& c, const std::vector<std::vector<T> >& a);
