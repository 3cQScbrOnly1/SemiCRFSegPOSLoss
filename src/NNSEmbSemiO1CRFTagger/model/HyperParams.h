#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{
public:
	int wordContext, wordWindow;
	int wordDim;
	vector<int> typeDims;
	int unitSize;
	vector<int> maxLabelLength;
	vector<int> maxcLabelLength;
	int segDim;
	int maxsegLen;
	dtype dropProb;
	int hiddenSize;
	int hiddenSize1;
	int rnnHiddenSize;
	int hiddenSize2;
	int segHiddenSize;
	int inputSize;	
	int labelSize;
	int clabelSize;
	// for optimization
	dtype nnRegular, adaAlpha, adaEps;
public:
	HyperParams(){
		bAssigned = false;
	}

	void setReqared(Options& opt){
		wordContext = opt.wordcontext;
		hiddenSize = opt.hiddenSize;
		hiddenSize1 = opt.hiddenSize;
		hiddenSize2 = opt.hiddenSize;
		segHiddenSize = opt.segHiddenSize;
		rnnHiddenSize = opt.rnnHiddenSize;
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
#endif /*SRC_HyperParams_H_ */
