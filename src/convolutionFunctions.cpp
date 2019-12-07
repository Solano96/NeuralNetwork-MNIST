#include "convolutionFunctions.h"

vector<vector<double> > vectorToMask(vector<double> &x){
	int msize = sqrt(x.size());

	assert(msize == sqrt(x.size()));

	vector<vector<double> > x_mask(msize, vector<double>(msize));

	for(int i = 0; i < msize; i++){
		for(int j = 0; j < msize; j++){
			x_mask[i][j] = x[msize*i+j];
		}
	}

	return x_mask;
}


vector<double> convolution(vector<double> &x, int h, int w, vector<vector<double> > mask, int msize){
	vector<double> x_convolution((h-msize+1)*(w-msize+1));

	for(int i = 0; i <= h-msize; i++){
		for(int j = 0; j <= w-msize; j++){
			double h = 0.0;
			for(int m_i = 0; m_i < msize; m_i++){
				for(int m_j = 0; m_j < msize; m_j++){
					h += x[w*(i+m_i)+(j+m_j)]*mask[m_i][m_j];
				}
			}
			x_convolution[(w-msize+1)*i + j] = h;
		}
	}
	return x_convolution;
}


vector<double> pooling(vector<double> &x, int h, int w, int psize){
	vector<double> x_pooling;

	for(int i = 0; i < h; i += psize){
		for(int j = 0; j < w; j += psize){
			double max = x[ i*w + j ];
			for(int m_i = 0; m_i < psize; m_i++){
				for(int m_j = 0; m_j < psize; m_j++){
					if( max < x[ (i+m_i)*w + (j+m_j) ] ){
						max = x[ (i+m_i)*w + (j+m_j) ];
					}
				}
			}
			x_pooling.push_back(max);
		}
	}
	return x_pooling;
}
