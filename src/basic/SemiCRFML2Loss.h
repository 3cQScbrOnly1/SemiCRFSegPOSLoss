#ifndef _SemiCRFML2LOSS_H_
#define _SemiCRFML2LOSS_H_

#include "N3L.h"

using namespace Eigen;

struct SemiCRFML2Loss{
public:
	int labelSize;
	int clabelSize;
	hash_map<int, int> f2c;
	vector<dtype> buffer;
	dtype eps;
	vector<int> maxLens;
	vector<int> maxcLens;
	int maxLen;
	Param T;
	Param CT;
	dtype lambda;


public:
	SemiCRFML2Loss(){
		labelSize = 0;
		clabelSize = 0;
		buffer.clear();
		eps = 1e-20;
		maxLens.clear();
		maxcLens.clear();
		f2c.clear();		
		maxLen = 0;
		lambda = 0.5;
	}

	~SemiCRFML2Loss(){
		labelSize = 0;
		clabelSize = 0;
		buffer.clear();
		maxLens.clear();
		maxcLens.clear();
		f2c.clear();		
		maxLen = 0;
		lambda = 0.5;
	}

public:
	inline void initial(const vector<int>& lens, const vector<int>& clens, hash_map<int, int>& f2c_trans, int maxLength, dtype lambdaPara, int seed = 0){
		labelSize = lens.size();
		clabelSize = clens.size();
		maxLen = maxLength;
		maxLens.resize(labelSize);
		for (int idx = 0; idx < labelSize; idx++){
			maxLens[idx] = lens[idx];
		}
		maxcLens.resize(clabelSize);
		for (int idx = 0; idx < clabelSize; idx++){
			maxcLens[idx] = clens[idx];
		}
		f2c.clear();
		for(int i = 0; i < labelSize; i++){
			f2c[i] = f2c_trans[i];
		}
		lambda = lambdaPara;	
		srand(seed);
		T.initial(labelSize, labelSize);
		CT.initial(clabelSize, clabelSize);
	}

	inline void exportAdaParams(ModelUpdate& ada){
		ada.addParam(&T);
		ada.addParam(&CT);
	}


public:
	// 
	inline dtype loss(const NRMat<PNode>& x, const NRMat<PNode>& cx, const vector<vector<vector<dtype> > >& answer, Metric& eval, Metric& ceval, int batchsize = 1){


		int seq_size = x.nrows();
		static int clabel;
		//int maxLength = x.ncols();
		vector<vector<vector<dtype> > > canswer;
		canswer.resize(answer.size());
		
		for(int idx = 0; idx < canswer.size(); idx++){
			canswer[idx].resize(answer[idx].size());
			for(int idy = 0; idy < canswer[idx].size(); idy++){
				canswer[idx][idy].resize(clabelSize);
				for(int idz = 0; idz < clabelSize; idz++){
					canswer[idx][idy][idz] = 0;
				}
				for(int idz = 0; idz < labelSize; idz++){
					clabel = f2c[idz];
					canswer[idx][idy][clabel] += answer[idx][idy][idz];
				}
			}			
		}
		
		dtype cost1 = floss(x, answer, eval, batchsize);
		dtype cost2 = closs(cx, canswer, ceval, batchsize);
		
		return cost1 + cost2;
	}

	//viterbi decode algorithm
	inline void predict(const NRMat<PNode>& x, const NRMat<PNode>& cx, NRMat<int>& y){
		/*
		assert(x.nrows() > 0 && x.ncols() == x.nrows());
		int nDim = x[0][0]->dim;
		if (labelSize != nDim) {
			std::cerr << "semi crf max likelihood predict error: labelSize size invalid" << std::endl;
			return;
		}*/

		int seq_size = x.nrows();
		static int clabeli, clabelj;

		NRMat3d<dtype> maxScores(seq_size, maxLen, labelSize);
		NRMat3d<int> maxLastLabels(seq_size, maxLen, labelSize);
		NRMat3d<int> maxLastStarts(seq_size, maxLen, labelSize);
		NRMat3d<int> maxLastDists(seq_size, maxLen, labelSize);

		maxScores = 0.0; maxLastLabels = -2; 
		maxLastStarts = -2; maxLastDists = -2;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				clabeli = f2c[i];
				for (int dist = 0; dist < seq_size - idx && dist < maxLens[i]; dist++) {
					// can be changed with probabilities in future work
					if (idx == 0) {
						maxScores[idx][dist][i] = x[idx][dist]->val(i, 0) * lambda + cx[idx][dist]->val(clabeli, 0) * (1 - lambda);
						maxLastLabels[idx][dist][i] = -1;
						maxLastStarts[idx][dist][i] = -1;
						maxLastDists[idx][dist][i] = -1;
					}
					else {
						int maxLastLabel = -1;
						int maxLastStart = -1;
						int LastDist = -1;
						dtype maxscore = 0.0;
						for (int j = 0; j < labelSize; ++j) {
							clabelj = f2c[j];
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxLens[j]; prevdist++) {
								dtype curScore = lambda * (T.val(j, i) + x[idx][dist]->val(i, 0)) 
								               + (1 - lambda) * (CT.val(clabelj, clabeli) + cx[idx][dist]->val(clabeli, 0)) 
								               + maxScores[idx - prevdist][prevdist - 1][j];
								if (maxLastLabel == -1 || curScore > maxscore){
									maxLastLabel = j;
									maxLastStart = idx - prevdist;
									LastDist = prevdist - 1;
									maxscore = curScore;
								}
							}
						}
						maxScores[idx][dist][i] = maxscore;
						maxLastLabels[idx][dist][i] = maxLastLabel;
						maxLastStarts[idx][dist][i] = maxLastStart;
						maxLastDists[idx][dist][i] = LastDist;
					}

				}
			}
		}

		// below zero denotes no such segment
		y.resize(seq_size, maxLen);
		y = -1;
		dtype maxFinalScore = 0.0;
		int maxFinalLabel = -1;
		int maxFinalStart = -1;
		int maxFinalDist = -1;
		for (int j = 0; j < labelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxLens[j]; dist++) {
				dtype curScore = maxScores[seq_size - dist][dist - 1][j];
				if (maxFinalLabel == -1 || curScore > maxFinalScore){
					maxFinalLabel = j;
					maxFinalStart = seq_size - dist;
					maxFinalDist = dist - 1;
					maxFinalScore = curScore;
				}
			}
		}

		y[maxFinalStart][maxFinalDist] = maxFinalLabel;

		while (1){
			int lastLabel = maxLastLabels[maxFinalStart][maxFinalDist][maxFinalLabel];
			int lastStart = maxLastStarts[maxFinalStart][maxFinalDist][maxFinalLabel];
			int lastDist = maxLastDists[maxFinalStart][maxFinalDist][maxFinalLabel];

			if (lastStart < 0){
				assert(maxFinalStart == 0);
				break;
			}

			y[lastStart][lastDist] = lastLabel;
			maxFinalLabel = lastLabel;
			maxFinalStart = lastStart;
			maxFinalDist = lastDist;
		}

	}

	inline dtype cost(const NRMat<PNode>& x, const NRMat<PNode>& cx, const vector<vector<vector<dtype> > >& answer, int batchsize = 1){
		/*
		assert(x.nrows() > 0 && x.ncols() == x.nrows() && x.nrows() == answer.size() && x.nrows() == answer[0].size());
		int nDim = x[0][0]->dim;
		if (labelSize != nDim || labelSize != answer[0][0].size()) {
			std::cerr << "semi crf max likelihood loss error: label size invalid" << std::endl;
			return -1.0;
		}*/

		int seq_size = x.nrows();
		static int clabel;
		//int maxLength = x.ncols();
		vector<vector<vector<dtype> > > canswer;
		canswer.resize(answer.size());
		
		for(int idx = 0; idx < canswer.size(); idx++){
			canswer[idx].resize(answer[idx].size());
			for(int idy = 0; idy < canswer[idx].size(); idy++){
				canswer[idx][idy].resize(clabelSize);
				for(int idz = 0; idz < clabelSize; idz++){
					canswer[idx][idy][idz] = 0;
				}
				for(int idz = 0; idz < labelSize; idz++){
					clabel = f2c[idz];
					canswer[idx][idy][clabel] += answer[idx][idy][idz];
				}
			}			
		}
		
		dtype cost1 = fcost(x, answer, batchsize);
		dtype cost2 = ccost(cx, canswer, batchsize);
		
		return cost1 + cost2;
	}
	
	
protected:
		
	// 
	inline dtype floss(const NRMat<PNode>& x, const vector<vector<vector<dtype> > >& answer, Metric& eval, int batchsize){
		/*
		assert(x.nrows() > 0 && x.ncols() == x.nrows() && x.nrows() == answer.size() && x.nrows() == answer[0].size());
		int nDim = x[0][0]->dim;
		if (labelSize != nDim || labelSize != answer[0][0].size()) {
			std::cerr << "semi crf max likelihood loss error: label size invalid" << std::endl;
			return -1.0;
		}*/

		int seq_size = x.nrows();
		//int maxLength = x.ncols();

		for (int idx = 0; idx < seq_size; idx++) {
			for (int dist = 0; dist < seq_size - idx && dist < maxLen; dist++) {
				if (x[idx][dist]->loss.size() == 0){
					x[idx][dist]->loss = Mat::Zero(labelSize, 1);
				}
			}
		}


		// comute alpha values, only the above parts are valid
		NRMat3d<dtype> alpha(seq_size, maxLen, labelSize);
		NRMat3d<dtype> alpha_answer(seq_size, maxLen, labelSize);
		alpha = 0.0; alpha_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxLens[i]; dist++) {
					// can be changed with probabilities in future work
					if (idx == 0) {
						alpha[idx][dist][i] = x[idx][dist]->val(i, 0);
						alpha_answer[idx][dist][i] = x[idx][dist]->val(i, 0) + log(answer[idx][dist][i] + eps);
					}
					else {
						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxLens[j]; prevdist++) {
								buffer.push_back(T.val(j, i) + x[idx][dist]->val(i, 0) + alpha[idx - prevdist][prevdist - 1][j]);
							}
						}
						alpha[idx][dist][i] = logsumexp(buffer);

						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxLens[j]; prevdist++) {
								buffer.push_back(T.val(j, i) + x[idx][dist]->val(i, 0) + alpha_answer[idx - prevdist][prevdist - 1][j]);
							}
						}
						alpha_answer[idx][dist][i] = logsumexp(buffer) + log(answer[idx][dist][i] + eps);
					}

				}
			}
		}

		// loss computation
		buffer.clear();
		for (int j = 0; j < labelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxLens[j]; dist++) {
				buffer.push_back(alpha[seq_size - dist][dist - 1][j]);
			}
		}
		dtype logZ = logsumexp(buffer);

		buffer.clear();
		for (int j = 0; j < labelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxLens[j]; dist++) {
				buffer.push_back(alpha_answer[seq_size - dist][dist - 1][j]);
			}
		}
		dtype logZ_answer = logsumexp(buffer);

		dtype cost = (logZ - logZ_answer) / batchsize;

		// comute belta values
		NRMat3d<dtype> belta(seq_size, maxLen, labelSize);
		NRMat3d<dtype> belta_answer(seq_size, maxLen, labelSize);
		belta = 0.0; belta_answer = 0.0;
		for (int idx = seq_size; idx > 0; idx--) {
			for (int i = 0; i < labelSize; ++i) {
				for (int dist = 1; dist <= idx && dist <= maxLens[i]; dist++) {
					if (idx == seq_size) {
						belta[idx - dist][dist - 1][i] = 0.0;
						belta_answer[idx - dist][dist - 1][i] = log(answer[idx - dist][dist - 1][i] + eps);
					}
					else {
						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							for (int nextdist = 0; nextdist < seq_size - idx && nextdist < maxLens[j]; nextdist++) {
								buffer.push_back(T.val(i, j) + x[idx][nextdist]->val(j, 0) + belta[idx][nextdist][j]);
							}
						}
						belta[idx - dist][dist - 1][i] = logsumexp(buffer);

						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							for (int nextdist = 0; nextdist < seq_size - idx && nextdist < maxLens[j]; nextdist++) {
								buffer.push_back(T.val(i, j) + x[idx][nextdist]->val(j, 0) + belta_answer[idx][nextdist][j]);
							}
						}
						belta_answer[idx - dist][dist - 1][i] = logsumexp(buffer) + log(answer[idx - dist][dist - 1][i] + eps);
					}
				}
			}
		}

		//compute margins
		NRMat3d<dtype> margin(seq_size, maxLen, labelSize);
		NRMat3d<dtype> margin_answer(seq_size, maxLen, labelSize);
		NRMat<dtype> trans(labelSize, labelSize);
		NRMat<dtype> trans_answer(labelSize, labelSize);
		margin = 0.0; margin_answer = 0.0;
		trans = 0.0; trans_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxLens[i]; dist++) {
					margin[idx][dist][i] = exp(alpha[idx][dist][i] + belta[idx][dist][i] - logZ);
					margin_answer[idx][dist][i] = exp(alpha_answer[idx][dist][i] + belta_answer[idx][dist][i] - logZ_answer);

					if (idx > 0) {
						for (int j = 0; j < labelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxLens[j]; prevdist++) {
								dtype logvalue = alpha[idx - prevdist][prevdist - 1][j] + x[idx][dist]->val(i, 0) + T.val(j, i) + belta[idx][dist][i] - logZ;
								trans[j][i] += exp(logvalue);
								logvalue = alpha_answer[idx - prevdist][prevdist - 1][j] + x[idx][dist]->val(i, 0) + T.val(j, i) + belta_answer[idx][dist][i] - logZ_answer;
								trans_answer[j][i] += exp(logvalue);
							}
						}
					}
				}
			}
		}

		//compute transition matrix losses
		for (int i = 0; i < labelSize; ++i) {
			for (int j = 0; j < labelSize; ++j) {
				T.grad(i, j) += (trans[i][j] - trans_answer[i][j]) * lambda / batchsize;
			}
		}

		//compute loss
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxLens[i]; dist++) {
					if (margin_answer[idx][dist][i] > 0.5){
						eval.overall_label_count++;
						if (margin[idx][dist][i] > 0.5){
							eval.correct_label_count++;
						}
					}
					x[idx][dist]->loss(i, 0) = (margin[idx][dist][i] - margin_answer[idx][dist][i]) * lambda / batchsize;
				}
			}
		}

		return cost * lambda;
	}	
	
	// 
	inline dtype closs(const NRMat<PNode>& x, const vector<vector<vector<dtype> > >& answer, Metric& eval, int batchsize){
		/*
		assert(x.nrows() > 0 && x.ncols() == x.nrows() && x.nrows() == answer.size() && x.nrows() == answer[0].size());
		int nDim = x[0][0]->dim;
		if (clabelSize != nDim || clabelSize != answer[0][0].size()) {
			std::cerr << "semi crf max likelihood loss error: label size invalid" << std::endl;
			return -1.0;
		}*/

		int seq_size = x.nrows();
		//int maxLength = x.ncols();

		for (int idx = 0; idx < seq_size; idx++) {
			for (int dist = 0; dist < seq_size - idx && dist < maxLen; dist++) {
				if (x[idx][dist]->loss.size() == 0){
					x[idx][dist]->loss = Mat::Zero(clabelSize, 1);
				}
			}
		}


		// comute alpha values, only the above parts are valid
		NRMat3d<dtype> alpha(seq_size, maxLen, clabelSize);
		NRMat3d<dtype> alpha_answer(seq_size, maxLen, clabelSize);
		alpha = 0.0; alpha_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < clabelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxcLens[i]; dist++) {
					// can be changed with probabilities in future work
					if (idx == 0) {
						alpha[idx][dist][i] = x[idx][dist]->val(i, 0);
						alpha_answer[idx][dist][i] = x[idx][dist]->val(i, 0) + log(answer[idx][dist][i] + eps);
					}
					else {
						buffer.clear();
						for (int j = 0; j < clabelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxcLens[j]; prevdist++) {
								buffer.push_back(CT.val(j, i) + x[idx][dist]->val(i, 0) + alpha[idx - prevdist][prevdist - 1][j]);
							}
						}
						alpha[idx][dist][i] = logsumexp(buffer);

						buffer.clear();
						for (int j = 0; j < clabelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxcLens[j]; prevdist++) {
								buffer.push_back(CT.val(j, i) + x[idx][dist]->val(i, 0) + alpha_answer[idx - prevdist][prevdist - 1][j]);
							}
						}
						alpha_answer[idx][dist][i] = logsumexp(buffer) + log(answer[idx][dist][i] + eps);
					}

				}
			}
		}

		// loss computation
		buffer.clear();
		for (int j = 0; j < clabelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxcLens[j]; dist++) {
				buffer.push_back(alpha[seq_size - dist][dist - 1][j]);
			}
		}
		dtype logZ = logsumexp(buffer);

		buffer.clear();
		for (int j = 0; j < clabelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxcLens[j]; dist++) {
				buffer.push_back(alpha_answer[seq_size - dist][dist - 1][j]);
			}
		}
		dtype logZ_answer = logsumexp(buffer);

		dtype cost = (logZ - logZ_answer) / batchsize;

		// comute belta values
		NRMat3d<dtype> belta(seq_size, maxLen, clabelSize);
		NRMat3d<dtype> belta_answer(seq_size, maxLen, clabelSize);
		belta = 0.0; belta_answer = 0.0;
		for (int idx = seq_size; idx > 0; idx--) {
			for (int i = 0; i < clabelSize; ++i) {
				for (int dist = 1; dist <= idx && dist <= maxcLens[i]; dist++) {
					if (idx == seq_size) {
						belta[idx - dist][dist - 1][i] = 0.0;
						belta_answer[idx - dist][dist - 1][i] = log(answer[idx - dist][dist - 1][i] + eps);
					}
					else {
						buffer.clear();
						for (int j = 0; j < clabelSize; ++j) {
							for (int nextdist = 0; nextdist < seq_size - idx && nextdist < maxcLens[j]; nextdist++) {
								buffer.push_back(CT.val(i, j) + x[idx][nextdist]->val(j, 0) + belta[idx][nextdist][j]);
							}
						}
						belta[idx - dist][dist - 1][i] = logsumexp(buffer);

						buffer.clear();
						for (int j = 0; j < clabelSize; ++j) {
							for (int nextdist = 0; nextdist < seq_size - idx && nextdist < maxcLens[j]; nextdist++) {
								buffer.push_back(CT.val(i, j) + x[idx][nextdist]->val(j, 0) + belta_answer[idx][nextdist][j]);
							}
						}
						belta_answer[idx - dist][dist - 1][i] = logsumexp(buffer) + log(answer[idx - dist][dist - 1][i] + eps);
					}
				}
			}
		}

		//compute margins
		NRMat3d<dtype> margin(seq_size, maxLen, clabelSize);
		NRMat3d<dtype> margin_answer(seq_size, maxLen, clabelSize);
		NRMat<dtype> trans(clabelSize, clabelSize);
		NRMat<dtype> trans_answer(clabelSize, clabelSize);
		margin = 0.0; margin_answer = 0.0;
		trans = 0.0; trans_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < clabelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxcLens[i]; dist++) {
					margin[idx][dist][i] = exp(alpha[idx][dist][i] + belta[idx][dist][i] - logZ);
					margin_answer[idx][dist][i] = exp(alpha_answer[idx][dist][i] + belta_answer[idx][dist][i] - logZ_answer);

					if (idx > 0) {
						for (int j = 0; j < clabelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxcLens[j]; prevdist++) {
								dtype logvalue = alpha[idx - prevdist][prevdist - 1][j] + x[idx][dist]->val(i, 0) + CT.val(j, i) + belta[idx][dist][i] - logZ;
								trans[j][i] += exp(logvalue);
								logvalue = alpha_answer[idx - prevdist][prevdist - 1][j] + x[idx][dist]->val(i, 0) + CT.val(j, i) + belta_answer[idx][dist][i] - logZ_answer;
								trans_answer[j][i] += exp(logvalue);
							}
						}
					}
				}
			}
		}

		//compute transition matrix losses
		for (int i = 0; i < clabelSize; ++i) {
			for (int j = 0; j < clabelSize; ++j) {
				CT.grad(i, j) += (trans[i][j] - trans_answer[i][j]) * (1 - lambda) / batchsize;
			}
		}

		//compute loss
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < clabelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxcLens[i]; dist++) {
					if (margin_answer[idx][dist][i] > 0.5){
						eval.overall_label_count++;
						if (margin[idx][dist][i] > 0.5){
							eval.correct_label_count++;
						}
					}
					x[idx][dist]->loss(i, 0) = (margin[idx][dist][i] - margin_answer[idx][dist][i]) * (1 - lambda) / batchsize;
				}
			}
		}

		return cost * (1 - lambda);
	}		

	inline dtype fcost(const NRMat<PNode>& x, const vector<vector<vector<dtype> > >& answer, int batchsize){
		/*
		assert(x.nrows() > 0 && x.ncols() == x.nrows() && x.nrows() == answer.size() && x.nrows() == answer[0].size());
		int nDim = x[0][0]->dim;
		if (labelSize != nDim || labelSize != answer[0][0].size()) {
			std::cerr << "semi crf max likelihood loss error: label size invalid" << std::endl;
			return -1.0;
		}*/

		int seq_size = x.nrows();
		// comute alpha values, only the above parts are valid
		NRMat3d<dtype> alpha(seq_size, maxLen, labelSize);
		NRMat3d<dtype> alpha_answer(seq_size, maxLen, labelSize);
		alpha = 0.0; alpha_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxLens[i]; dist++) {
					// can be changed with probabilities in future work
					if (idx == 0) {
						alpha[idx][dist][i] = x[idx][dist]->val(i, 0);
						alpha_answer[idx][dist][i] = x[idx][dist]->val(i, 0) + log(answer[idx][dist][i] + eps);
					}
					else {
						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxLens[j]; prevdist++) {
								buffer.push_back(T.val(j, i) + x[idx][dist]->val(i, 0) + alpha[idx - prevdist][prevdist - 1][j]);
							}
						}
						alpha[idx][dist][i] = logsumexp(buffer);

						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxLens[j]; prevdist++) {
								buffer.push_back(T.val(j, i) + x[idx][dist]->val(i, 0) + alpha_answer[idx - prevdist][prevdist - 1][j]);
							}
						}
						alpha_answer[idx][dist][i] = logsumexp(buffer) + log(answer[idx][dist][i] + eps);
					}

				}
			}
		}

		// loss computation
		buffer.clear();
		for (int j = 0; j < labelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxLens[j]; dist++) {
				buffer.push_back(alpha[seq_size - dist][dist - 1][j]);
			}
		}
		dtype logZ = logsumexp(buffer);

		buffer.clear();
		for (int j = 0; j < labelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxLens[j]; dist++) {
				buffer.push_back(alpha_answer[seq_size - dist][dist - 1][j]);
			}
		}
		dtype logZ_answer = logsumexp(buffer);

		return (logZ - logZ_answer) * lambda / batchsize;
	}

	inline dtype ccost(const NRMat<PNode>& x, const vector<vector<vector<dtype> > >& answer, int batchsize){
		/*
		assert(x.nrows() > 0 && x.ncols() == x.nrows() && x.nrows() == answer.size() && x.nrows() == answer[0].size());
		int nDim = x[0][0]->dim;
		if (clabelSize != nDim || clabelSize != answer[0][0].size()) {
			std::cerr << "semi crf max likelihood loss error: label size invalid" << std::endl;
			return -1.0;
		}*/

		int seq_size = x.nrows();
		// comute alpha values, only the above parts are valid
		NRMat3d<dtype> alpha(seq_size, maxLen, clabelSize);
		NRMat3d<dtype> alpha_answer(seq_size, maxLen, clabelSize);
		alpha = 0.0; alpha_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < clabelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxcLens[i]; dist++) {
					// can be changed with probabilities in future work
					if (idx == 0) {
						alpha[idx][dist][i] = x[idx][dist]->val(i, 0);
						alpha_answer[idx][dist][i] = x[idx][dist]->val(i, 0) + log(answer[idx][dist][i] + eps);
					}
					else {
						buffer.clear();
						for (int j = 0; j < clabelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxcLens[j]; prevdist++) {
								buffer.push_back(CT.val(j, i) + x[idx][dist]->val(i, 0) + alpha[idx - prevdist][prevdist - 1][j]);
							}
						}
						alpha[idx][dist][i] = logsumexp(buffer);

						buffer.clear();
						for (int j = 0; j < clabelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxcLens[j]; prevdist++) {
								buffer.push_back(CT.val(j, i) + x[idx][dist]->val(i, 0) + alpha_answer[idx - prevdist][prevdist - 1][j]);
							}
						}
						alpha_answer[idx][dist][i] = logsumexp(buffer) + log(answer[idx][dist][i] + eps);
					}

				}
			}
		}

		// loss computation
		buffer.clear();
		for (int j = 0; j < clabelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxcLens[j]; dist++) {
				buffer.push_back(alpha[seq_size - dist][dist - 1][j]);
			}
		}
		dtype logZ = logsumexp(buffer);

		buffer.clear();
		for (int j = 0; j < clabelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxcLens[j]; dist++) {
				buffer.push_back(alpha_answer[seq_size - dist][dist - 1][j]);
			}
		}
		dtype logZ_answer = logsumexp(buffer);

		return (logZ - logZ_answer) * (1 - lambda) / batchsize;
	}	
		
};


#endif /* _SemiCRFML2LOSS_H_ */
