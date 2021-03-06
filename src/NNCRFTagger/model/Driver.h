#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include <iostream>
#include "ComputionGraph.h"

class Driver {
public:
	Driver(int memsize) :_aligned_mem(memsize){
		_pcg = NULL;
	}

	~Driver() {
		if (_pcg != NULL)
			delete _pcg;
		_pcg = NULL;
	}

public:
	ModelParams _model_params;
	HyperParams _hyper_params;

	Metric _eval;


	ModelUpdate _ada;

	ComputionGraph *_pcg;

	CheckGrad _checkgrad;

	AlignedMemoryPool _aligned_mem;

public:
	//embeddings are initialized before this separately.
	inline void initial(){
		if (!_hyper_params.bVaild()){
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}

		if (!_model_params.initial(_hyper_params)){
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_model_params.exportModelParams(_ada);
		_model_params.exportCheckGradParams(_checkgrad);
		_hyper_params.print();

		_pcg = new ComputionGraph();
		_pcg->createNodes(ComputionGraph::max_sentence_length, _model_params._types.size());
		_pcg->initial(_model_params, _hyper_params, &_aligned_mem);

		setUpdateParameters(_hyper_params.nnRegular, _hyper_params.adaAlpha, _hyper_params.adaEps);

	}


	inline dtype train(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			//forward
			_pcg->forward(example.m_features, true);

			//loss function
			int seq_size = example.m_features.size();
			cost += _model_params._loss.loss(getPNodes(_pcg->output, seq_size), getPNodes(_pcg->coutput, seq_size), example.m_labels, _eval, _eval, example_num);

			// backward, which exists only for training 
			_pcg->backward();
		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	inline void predict(const vector<Feature>& features, vector<int>& results) {
		_pcg->forward(features);
		int seq_size = features.size();
		//results.resize(seq_size);
		_model_params._loss.predict(getPNodes(_pcg->output, seq_size), getPNodes(_pcg->coutput, seq_size), results);
	}

	inline dtype cost(const Example& example){
		_pcg->forward(example.m_features); //forward here

		int seq_size = example.m_features.size();

		dtype cost = 0.0;
		//loss function
		cost += _model_params._loss.cost(getPNodes(_pcg->output, seq_size), getPNodes(_pcg->coutput, seq_size), example.m_labels, 1);

		return cost;
	}


	void updateModel() {
		//_ada.update();
		_ada.update(5.0);
	}

	void checkgrad(const vector<Example>& examples, int iter){
		ostringstream out;
		out << "Iteration: " << iter;
		_checkgrad.check(this, examples, out.str());
	}

	void writeModel();

	void loadModel();



public:
	inline void resetEval() {
		_eval.reset();
	}


	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

#endif /*SRC_Driver_H_ */
