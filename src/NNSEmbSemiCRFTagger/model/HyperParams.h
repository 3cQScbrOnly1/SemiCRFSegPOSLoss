#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{
	int wordContext, wordWindow;
	int wordDim;
	vector<int> typeDims;
	int unitSize;
	int segDim;

	dtype dropProb;
	int hiddenSize1;
	int hiddenSize2;
	int rnnHiddenSize;
	int segHiddenSize;
	int inputSize;
	int labelSize;
	int clabelSize;
	int maxsegLen;

	vector<int> maxLabelLength;
	vector<int> maxcLabelLength;
	// for optimization
	dtype nnRegular, adaAlpha, adaEps;
public:
	HyperParams(){
		bAssigned = false;
	}

	void setReqared(Options& opt){
		wordContext = opt.wordcontext;
		hiddenSize1 = opt.hiddenSize;
		hiddenSize2 = opt.hiddenSize;
		rnnHiddenSize = opt.rnnHiddenSize;
		segHiddenSize = opt.segHiddenSize;
		dropProb = opt.dropProb;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bVaild(){
		return bAssigned;
	}

	void print(){}

private:
	bool bAssigned;
};

#endif /*SRC_HyperParams_H_*/
