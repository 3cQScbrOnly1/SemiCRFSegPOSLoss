#ifndef _CRFML2LOSS_H_
#define _CRFML2LOSS_H_

#include "N3L.h"

using namespace Eigen;

struct CRFML2Loss{
public:
	Param T;
	Param CT;
	int labelSize;
	int clabelSize;
	unordered_map<int, int> f2c;
	vector<dtype> buffer;
	dtype eps;
	dtype lambda; // lambda * fine  + (1-lambda) * coarse

public:
	CRFML2Loss(){
		labelSize = 0;
		clabelSize = 0;
		f2c.clear();
		buffer.clear();
		eps = 1e-20;
		lambda = 0.5;
	}

	~CRFML2Loss(){
		labelSize = 0;
		clabelSize = 0;
		buffer.clear();
		f2c.clear();
	}

public:
	inline void initial(int labelNum, int clabelNum, unordered_map<int, int>& f2c_trans, dtype lambdaPara, int seed = 0){
		srand(seed);
		labelSize = labelNum;
		clabelSize = clabelNum;
		T.initial(labelSize, labelSize);
		CT.initial(clabelSize, clabelSize);
		f2c.clear();
		for(int i = 0; i < labelSize; i++){
			f2c[i] = f2c_trans[i];
		}
		lambda = lambdaPara;
	}

	inline void exportAdaParams(ModelUpdate& ada){
		ada.addParam(&T);
		ada.addParam(&CT);
	}

public:
	inline dtype loss(const vector<PNode>& x, const vector<PNode>& cx, const vector<vector<dtype> >&answer, Metric& eval, Metric& ceval, int batchsize = 1){
		assert(x.size() > 0);
		vector<vector<dtype> > canswer;
		int seq_size = answer.size();
		canswer.resize(answer.size());
		static int clabel;
		for (int idx = 0; idx < seq_size; idx++){
			canswer[idx].resize(clabelSize);
			for(int idy = 0; idy < clabelSize; idy++){
				canswer[idx][idy] = 0;
			}
			for(int idy = 0; idy < labelSize; idy++){
				clabel = f2c[idy];
				canswer[idx][clabel] += answer[idx][idy];
			}
		}
		
		dtype cost1 = floss(x, answer, eval, batchsize);
		dtype cost2 = closs(cx, canswer, ceval, batchsize);
		
		return cost1 + cost2;
		
	}

	//viterbi decode algorithm
	inline void predict(const vector<PNode>& x, const vector<PNode>& cx, vector<int>& y){
		assert(x.size() > 0);
		int nDim = x[0]->dim;
		int nCDim = cx[0]->dim;
		if (labelSize != nDim || clabelSize != nCDim) {
			std::cerr << "crf max likelihood predict error: dim size invalid" << std::endl;
			return;
		}
		
		buffer.resize(labelSize);

		int seq_size = x.size();
		static int clabeli, clabelj;

		NRMat<dtype> maxScores(seq_size, labelSize);
		NRMat<int> maxLastLabels(seq_size, labelSize);
		maxScores = 0.0; maxLastLabels = -2;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				// can be changed with probabilities in future work
				clabeli = f2c[i];
				if (idx == 0) {
					maxScores[idx][i] = lambda * x[idx]->val(i, 0) + (1 - lambda) * cx[idx]->val(clabeli, 0);
					maxLastLabels[idx][i] = -1;
				}
				else {
					int maxLastLabel = -1;
					dtype maxscore = 0.0;
					for (int j = 0; j < labelSize; ++j) {
						clabelj = f2c[j];
						dtype curscore = lambda * (T.val(j, i) + x[idx]->val(i, 0)) + (1 - lambda) * (CT.val(clabelj, clabeli) + cx[idx]->val(clabeli, 0)) + maxScores[idx - 1][j];
						if (maxLastLabel == -1 || curscore > maxscore) {
							maxLastLabel = j;
							maxscore = curscore;
						}
					}
					maxScores[idx][i] = maxscore;
					maxLastLabels[idx][i] = maxLastLabel;
				}
			}
		}

		y.resize(seq_size);
		dtype maxFinalScore = maxScores[seq_size - 1][0];
		y[seq_size - 1] = 0;
		for (int i = 1; i < labelSize; ++i) {
			if (maxScores[seq_size - 1][i] > maxFinalScore) {
				maxFinalScore = maxScores[seq_size - 1][i];
				y[seq_size - 1] = i;
			}
		}

		for (int idx = seq_size - 2; idx >= 0; idx--) {
			y[idx] = maxLastLabels[idx + 1][y[idx + 1]];
		}

	}


inline dtype cost(const vector<PNode>& x, const vector<PNode>& cx, const vector<vector<dtype> >&answer, int batchsize = 1){
		assert(x.size() > 0);
		vector<vector<dtype> > canswer;
		int seq_size = answer.size();
		canswer.resize(answer.size());
		static int clabel;
		for (int idx = 0; idx < seq_size; idx++){
			canswer[idx].resize(clabelSize);
			for(int idy = 0; idy < clabelSize; idy++){
				canswer[idx][idy] = 0;
			}
			for(int idy = 0; idy < labelSize; idy++){
				clabel = f2c[idy];
				canswer[idx][clabel] += answer[idx][idy];
			}
		}
		
		dtype cost1 = fcost(x, answer, batchsize);
		dtype cost2 = ccost(cx, canswer, batchsize);
		
		return cost1 + cost2;
}


protected:	
	inline dtype floss(const vector<PNode>& x, const vector<vector<dtype> >&answer, Metric& eval, int batchsize){
		assert(x.size() > 0 && x.size() == answer.size());
		int nDim = x[0]->dim;
		if (labelSize != nDim || labelSize != answer[0].size()) {
			std::cerr << "crf max likelihood loss error: dim size invalid" << std::endl;
			return -1.0;
		}
		int seq_size = x.size();
		for (int idx = 0; idx < seq_size; idx++){
			if (x[idx]->loss.size() == 0){
				x[idx]->loss = Mat::Zero(labelSize, 1);
			}
		}
		
		buffer.resize(labelSize);

		// comute alpha values
		NRMat<dtype> alpha(seq_size, labelSize);
		NRMat<dtype> alpha_answer(seq_size, labelSize);
		alpha = 0.0; alpha_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				// can be changed with probabilities in future work
				if (idx == 0) {
					alpha[idx][i] = x[idx]->val(i, 0);
					alpha_answer[idx][i] = x[idx]->val(i, 0) + log(answer[idx][i] + eps);
				}
				else {
					for (int j = 0; j < labelSize; ++j) {
						buffer[j] = T.val(j, i) + x[idx]->val(i, 0) + alpha[idx - 1][j];
					}
					alpha[idx][i] = logsumexp(buffer);

					for (int j = 0; j < labelSize; ++j) {
						buffer[j] = T.val(j, i) + x[idx]->val(i, 0) + alpha_answer[idx - 1][j];
					}
					alpha_answer[idx][i] = logsumexp(buffer) + log(answer[idx][i] + eps);
				}
			}
		}

		// loss computation
		for (int j = 0; j < labelSize; ++j) {
			buffer[j] = alpha[seq_size - 1][j];
		}
		dtype logZ = logsumexp(buffer);

		for (int j = 0; j < labelSize; ++j) {
			buffer[j] = alpha_answer[seq_size - 1][j];
		}
		dtype logZ_answer = logsumexp(buffer);
		dtype cost = (logZ - logZ_answer) / batchsize;

		// comute belta values
		NRMat<dtype> belta(seq_size, labelSize);
		NRMat<dtype> belta_answer(seq_size, labelSize);
		belta = 0.0; belta_answer = 0.0;
		for (int idx = seq_size - 1; idx >= 0; idx--) {
			for (int i = 0; i < labelSize; ++i) {
				if (idx == seq_size - 1) {
					belta[idx][i] = 0.0;
					belta_answer[idx][i] = log(answer[idx][i] + eps);
				}
				else {
					for (int j = 0; j < labelSize; ++j) {
						buffer[j] = T.val(i, j) + x[idx + 1]->val(j, 0) + belta[idx + 1][j];
					}
					belta[idx][i] = logsumexp(buffer);

					for (int j = 0; j < labelSize; ++j) {
						buffer[j] = T.val(i, j) + x[idx + 1]->val(j, 0) + belta_answer[idx + 1][j];
					}
					belta_answer[idx][i] = logsumexp(buffer) + log(answer[idx][i] + eps);
				}
			}
		}

		NRMat<dtype> margin(seq_size, labelSize);
		NRMat<dtype> trans(labelSize, labelSize);
		NRMat<dtype> margin_answer(seq_size, labelSize);
		NRMat<dtype> trans_answer(labelSize, labelSize);
		margin = 0.0; trans = 0.0; margin_answer = 0.0; trans_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			dtype sum = 0.0, sum_answer = 0.0;
			for (int i = 0; i < labelSize; ++i) {
				margin[idx][i] = exp(alpha[idx][i] + belta[idx][i] - logZ);
				margin_answer[idx][i] = exp(alpha_answer[idx][i] + belta_answer[idx][i] - logZ_answer);
				if (idx > 0) {
					for (int j = 0; j < labelSize; ++j) {
						dtype logvalue = alpha[idx - 1][j] + x[idx]->val(i, 0) + T.val(j, i) + belta[idx][i] - logZ;
						trans[j][i] += exp(logvalue);
						logvalue = alpha_answer[idx - 1][j] + x[idx]->val(i, 0) + T.val(j, i) + belta_answer[idx][i] - logZ_answer;
						trans_answer[j][i] += exp(logvalue);
					}
				}

				sum += margin[idx][i];
				sum_answer += margin_answer[idx][i];
			}
			if (abs(sum - 1) > 1e-6 || abs(sum_answer - 1) > 1e-6){
				std::cout << "prob sum:  free = " << sum << ", answer = " << sum_answer << std::endl;
			}
		}

		//compute transition matrix losses
		for (int i = 0; i < labelSize; ++i) {
			for (int j = 0; j < labelSize; ++j) {
				T.grad(i, j) += (trans[i][j] - trans_answer[i][j]) * lambda / batchsize;
			}
		}

		//compute 
		eval.overall_label_count += seq_size;

		for (int idx = 0; idx < seq_size; idx++) {
			int bestid = -1, bestid_answer = -1;
			for (int i = 0; i < labelSize; ++i) {
				x[idx]->loss(i, 0) = (margin[idx][i] - margin_answer[idx][i]) * lambda / batchsize;
				if (bestid == -1 || margin[idx][i] > margin[idx][bestid]) {
					bestid = i;
				}
				if (bestid_answer == -1 || margin_answer[idx][i] > margin_answer[idx][bestid_answer]) {
					bestid_answer = i;
				}
			}

			if (bestid != -1 && bestid == bestid_answer)
				eval.correct_label_count++;

			if (bestid == -1){
				std::cout << "error, please debug" << std::endl;
			}

		}

		return cost * lambda;
	}
	
	inline dtype closs(const vector<PNode>& x, const vector<vector<dtype> >&answer, Metric& eval, int batchsize){
		assert(x.size() > 0 && x.size() == answer.size());
		int nDim = x[0]->dim;
		if (clabelSize != nDim || clabelSize != answer[0].size()) {
			std::cerr << "crf max likelihood loss error: dim size invalid" << std::endl;
			return -1.0;
		}
		int seq_size = x.size();
		for (int idx = 0; idx < seq_size; idx++){
			if (x[idx]->loss.size() == 0){
				x[idx]->loss = Mat::Zero(clabelSize, 1);
			}
		}
		
		buffer.resize(clabelSize);

		// comute alpha values
		NRMat<dtype> alpha(seq_size, clabelSize);
		NRMat<dtype> alpha_answer(seq_size, clabelSize);
		alpha = 0.0; alpha_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < clabelSize; ++i) {
				// can be changed with probabilities in future work
				if (idx == 0) {
					alpha[idx][i] = x[idx]->val(i, 0);
					alpha_answer[idx][i] = x[idx]->val(i, 0) + log(answer[idx][i] + eps);
				}
				else {
					for (int j = 0; j < clabelSize; ++j) {
						buffer[j] = CT.val(j, i) + x[idx]->val(i, 0) + alpha[idx - 1][j];
					}
					alpha[idx][i] = logsumexp(buffer);

					for (int j = 0; j < clabelSize; ++j) {
						buffer[j] = CT.val(j, i) + x[idx]->val(i, 0) + alpha_answer[idx - 1][j];
					}
					alpha_answer[idx][i] = logsumexp(buffer) + log(answer[idx][i] + eps);
				}
			}
		}

		// loss computation
		for (int j = 0; j < clabelSize; ++j) {
			buffer[j] = alpha[seq_size - 1][j];
		}
		dtype logZ = logsumexp(buffer);

		for (int j = 0; j < clabelSize; ++j) {
			buffer[j] = alpha_answer[seq_size - 1][j];
		}
		dtype logZ_answer = logsumexp(buffer);
		dtype cost = (logZ - logZ_answer) / batchsize;

		// comute belta values
		NRMat<dtype> belta(seq_size, clabelSize);
		NRMat<dtype> belta_answer(seq_size, clabelSize);
		belta = 0.0; belta_answer = 0.0;
		for (int idx = seq_size - 1; idx >= 0; idx--) {
			for (int i = 0; i < clabelSize; ++i) {
				if (idx == seq_size - 1) {
					belta[idx][i] = 0.0;
					belta_answer[idx][i] = log(answer[idx][i] + eps);
				}
				else {
					for (int j = 0; j < clabelSize; ++j) {
						buffer[j] = CT.val(i, j) + x[idx + 1]->val(j, 0) + belta[idx + 1][j];
					}
					belta[idx][i] = logsumexp(buffer);

					for (int j = 0; j < clabelSize; ++j) {
						buffer[j] = CT.val(i, j) + x[idx + 1]->val(j, 0) + belta_answer[idx + 1][j];
					}
					belta_answer[idx][i] = logsumexp(buffer) + log(answer[idx][i] + eps);
				}
			}
		}

		NRMat<dtype> margin(seq_size, clabelSize);
		NRMat<dtype> trans(clabelSize, clabelSize);
		NRMat<dtype> margin_answer(seq_size, clabelSize);
		NRMat<dtype> trans_answer(clabelSize, clabelSize);
		margin = 0.0; trans = 0.0; margin_answer = 0.0; trans_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			dtype sum = 0.0, sum_answer = 0.0;
			for (int i = 0; i < clabelSize; ++i) {
				margin[idx][i] = exp(alpha[idx][i] + belta[idx][i] - logZ);
				margin_answer[idx][i] = exp(alpha_answer[idx][i] + belta_answer[idx][i] - logZ_answer);
				if (idx > 0) {
					for (int j = 0; j < clabelSize; ++j) {
						dtype logvalue = alpha[idx - 1][j] + x[idx]->val(i, 0) + CT.val(j, i) + belta[idx][i] - logZ;
						trans[j][i] += exp(logvalue);
						logvalue = alpha_answer[idx - 1][j] + x[idx]->val(i, 0) + CT.val(j, i) + belta_answer[idx][i] - logZ_answer;
						trans_answer[j][i] += exp(logvalue);
					}
				}

				sum += margin[idx][i];
				sum_answer += margin_answer[idx][i];
			}
			if (abs(sum - 1) > 1e-6 || abs(sum_answer - 1) > 1e-6){
				std::cout << "prob sum:  free = " << sum << ", answer = " << sum_answer << std::endl;
			}
		}

		//compute transition matrix losses
		for (int i = 0; i < clabelSize; ++i) {
			for (int j = 0; j < clabelSize; ++j) {
				CT.grad(i, j) += (trans[i][j] - trans_answer[i][j]) * (1 - lambda) / batchsize;
			}
		}

		//compute 
		eval.overall_label_count += seq_size;

		for (int idx = 0; idx < seq_size; idx++) {
			int bestid = -1, bestid_answer = -1;
			for (int i = 0; i < clabelSize; ++i) {
				x[idx]->loss(i, 0) = (margin[idx][i] - margin_answer[idx][i]) * (1 - lambda) / batchsize;
				if (bestid == -1 || margin[idx][i] > margin[idx][bestid]) {
					bestid = i;
				}
				if (bestid_answer == -1 || margin_answer[idx][i] > margin_answer[idx][bestid_answer]) {
					bestid_answer = i;
				}
			}

			if (bestid != -1 && bestid == bestid_answer)
				eval.correct_label_count++;

			if (bestid == -1){
				std::cout << "error, please debug" << std::endl;
			}

		}

		return cost * (1 - lambda);
	}	
	
	
	
inline dtype fcost(const vector<PNode>& x, const vector<vector<dtype> >&answer, int batchsize){
		assert(x.size() > 0 && x.size() == answer.size());
		int nDim = x[0]->dim;
		if (labelSize != nDim || labelSize != answer[0].size()) {
			std::cerr << "crf max likelihood cost error: dim size invalid" << std::endl;
			return -1.0;
		}

		buffer.resize(labelSize);
		int seq_size = x.size();
		// comute alpha values
		NRMat<dtype> alpha(seq_size, labelSize);
		NRMat<dtype> alpha_answer(seq_size, labelSize);
		alpha = 0.0; alpha_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				// can be changed with probabilities in future work
				if (idx == 0) {
					alpha[idx][i] = x[idx]->val(i, 0);
					alpha_answer[idx][i] = x[idx]->val(i, 0) + log(answer[idx][i] + eps);
				}
				else {
					for (int j = 0; j < labelSize; ++j) {
						buffer[j] = T.val(j, i) + x[idx]->val(i, 0) + alpha[idx - 1][j];
					}
					alpha[idx][i] = logsumexp(buffer);

					for (int j = 0; j < labelSize; ++j) {
						buffer[j] = T.val(j, i) + x[idx]->val(i, 0) + alpha_answer[idx - 1][j];
					}
					alpha_answer[idx][i] = logsumexp(buffer) + log(answer[idx][i] + eps);
				}
			}
		}

		// loss computation
		for (int j = 0; j < labelSize; ++j) {
			buffer[j] = alpha[seq_size - 1][j];
		}
		dtype logZ = logsumexp(buffer);

		for (int j = 0; j < labelSize; ++j) {
			buffer[j] = alpha_answer[seq_size - 1][j];
		}
		dtype logZ_answer = logsumexp(buffer);

		return (logZ - logZ_answer) * lambda / batchsize;
	}
	
	
	inline dtype ccost(const vector<PNode>& x, const vector<vector<dtype> >&answer, int batchsize){
		assert(x.size() > 0 && x.size() == answer.size());
		int nDim = x[0]->dim;
		if (clabelSize != nDim || clabelSize != answer[0].size()) {
			std::cerr << "crf max likelihood cost error: dim size invalid" << std::endl;
			return -1.0;
		}
				
		buffer.resize(clabelSize);
		int seq_size = x.size();

		// comute alpha values
		NRMat<dtype> alpha(seq_size, clabelSize);
		NRMat<dtype> alpha_answer(seq_size, clabelSize);
		alpha = 0.0; alpha_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < clabelSize; ++i) {
				// can be changed with probabilities in future work
				if (idx == 0) {
					alpha[idx][i] = x[idx]->val(i, 0);
					alpha_answer[idx][i] = x[idx]->val(i, 0) + log(answer[idx][i] + eps);
				}
				else {
					for (int j = 0; j < clabelSize; ++j) {
						buffer[j] = CT.val(j, i) + x[idx]->val(i, 0) + alpha[idx - 1][j];
					}
					alpha[idx][i] = logsumexp(buffer);

					for (int j = 0; j < clabelSize; ++j) {
						buffer[j] = CT.val(j, i) + x[idx]->val(i, 0) + alpha_answer[idx - 1][j];
					}
					alpha_answer[idx][i] = logsumexp(buffer) + log(answer[idx][i] + eps);
				}
			}
		}

		// loss computation
		for (int j = 0; j < clabelSize; ++j) {
			buffer[j] = alpha[seq_size - 1][j];
		}
		dtype logZ = logsumexp(buffer);

		for (int j = 0; j < clabelSize; ++j) {
			buffer[j] = alpha_answer[seq_size - 1][j];
		}
		dtype logZ_answer = logsumexp(buffer);
		
		return (logZ - logZ_answer) * (1 - lambda) / batchsize;
	}

};


#endif /* _CRFML2LOSS_H_ */
