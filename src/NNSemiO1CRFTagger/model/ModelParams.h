#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"
#include "BMESSegmentation.h"
#include "SemiCRFML2Loss.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{
public:
	LSTM1Params _left_lstm_project; //left lstm
	LSTM1Params _right_lstm_project; //right lstm
	UniParams _tanh1_project; // hidden
	BiParams _tanh2_project; // hidden
	SegParams _seglayer_project; //segmentation
	UniParams _olayer_linear; // output
	UniParams _colayer_linear; // output


	SemiCRFML2Loss _loss;

public:
	LookupTable _words;
	vector<LookupTable> _types;
	vector<Alphabet> _type_alphas;
	Alphabet _word_alpha;
	Alphabet _label_alpha;
	Alphabet _clabel_alpha;
	Alphabet _seg_label_alpha;
	Alphabet _cseg_label_alpha;
public:
	bool initial(HyperParams& hyper_params){
		if (_words.nVSize <= 0){
			std::cout << "Please initialize embeddings before this." << std::endl;
			return false;
		}
		hyper_params.wordWindow = hyper_params.wordContext * 2 + 1;
		hyper_params.wordDim = _words.nDim;
		hyper_params.unitSize = hyper_params.wordDim;
		hyper_params.typeDims.clear();
		for (int idx = 0; idx < _types.size(); idx++)
		{
			if (_types[idx].nVSize <= 0)
				return false;
			hyper_params.typeDims.push_back(_types[idx].nDim);
			hyper_params.unitSize += hyper_params.typeDims[idx];
		}

		hyper_params.labelSize = _loss.labelSize;
		hyper_params.clabelSize = _loss.clabelSize;
		hyper_params.inputSize = hyper_params.wordWindow * hyper_params.unitSize;
		_tanh1_project.initial(hyper_params.hiddenSize1, hyper_params.inputSize, true);
		_left_lstm_project.initial(hyper_params.rnnHiddenSize, hyper_params.hiddenSize1);
		_right_lstm_project.initial(hyper_params.rnnHiddenSize, hyper_params.hiddenSize1);
		_tanh2_project.initial(hyper_params.hiddenSize1, hyper_params.rnnHiddenSize, hyper_params.rnnHiddenSize, true);
		_seglayer_project.initial(hyper_params.segHiddenSize, hyper_params.hiddenSize2, hyper_params.hiddenSize1);
		_olayer_linear.initial(hyper_params.labelSize, hyper_params.segHiddenSize, false);
		_colayer_linear.initial(hyper_params.clabelSize, hyper_params.segHiddenSize, false);
	}

	void exportModelParams(ModelUpdate& ada){
		_words.exportAdaParams(ada);
		for (int idx = 0; idx < _types.size(); idx++)
			_types[idx].exportAdaParams(ada);
		_tanh1_project.exportAdaParams(ada);
		_left_lstm_project.exportAdaParams(ada);
		_right_lstm_project.exportAdaParams(ada);
		_tanh2_project.exportAdaParams(ada);
		_seglayer_project.exportAdaParams(ada);
		_olayer_linear.exportAdaParams(ada);
		_colayer_linear.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&(_words.E), "_words.E");
		for (int idx = 0; idx < _types.size(); idx++){
			stringstream ss;
			ss << "_types[" << idx << "].E";
			checkgrad.add(&(_types[idx].E), ss.str());
		}
		checkgrad.add(&(_tanh1_project.W), "_tan1_project.W");
		checkgrad.add(&(_tanh1_project.b), "_tan1_project.b");

		checkgrad.add(&(_tanh2_project.W1), "_tan1_project.W1");
		checkgrad.add(&(_tanh2_project.W2), "_tan1_project.W2");
		checkgrad.add(&(_tanh2_project.b), "_tan2_project.b");
	}

	void saveModel(){
	}
	void loadModel(const string& infile){
	}
};

#endif /* SRC_ModelParams_H_ */