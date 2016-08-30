/*
 * Tagger.cpp
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#include "NNSemiCRFTagger.h"

#include "Argument_helper.h"

Tagger::Tagger() {
	// TODO Auto-generated constructor stub
}

Tagger::~Tagger() {
	// TODO Auto-generated destructor stub
}

int Tagger::createAlphabet(const vector<Instance>& vecInsts) {
	if (vecInsts.size() == 0){
		std::cout << "training set empty" << std::endl;
		return -1;
	}
	cout << "Creating Alphabet..." << endl;

	int numInstance;

	m_labelAlphabet.clear();
	m_clabelAlphabet.clear();
	m_seglabelAlphabet.clear();
	m_cseglabelAlphabet.clear();
	ignoreLabels.clear();


	int typeNum = vecInsts[0].typefeatures[0].size();
	m_type_stats.resize(typeNum);

	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->words;
		const vector<string> &labels = pInstance->labels;
		const vector<vector<string> > &sparsefeatures = pInstance->sparsefeatures;
		const vector<vector<string> > &charfeatures = pInstance->charfeatures;

		const vector<vector<string> > &typefeatures = pInstance->typefeatures;
		for (int iter_type = 0; iter_type < typefeatures.size(); iter_type++) {
			assert(typeNum == typefeatures[iter_type].size());
		}

		int curInstSize = labels.size();
		int labelId, clabelId;
		for (int i = 0; i < curInstSize; ++i) {
			if (is_start_label(labels[i])){
				labelId = m_seglabelAlphabet.from_string(labels[i].substr(2));
				clabelId = m_cseglabelAlphabet.from_string("seg");
				m_seg_f2c[labelId] = clabelId;
			}
			else if (labels[i].length() == 1) {
				// usually O or o, trick
				labelId = m_seglabelAlphabet.from_string(labels[i]);
				clabelId = m_cseglabelAlphabet.from_string("seg");
				m_seg_f2c[labelId] = clabelId;
				ignoreLabels.insert(labels[i]);
			}
			labelId = m_labelAlphabet.from_string(labels[i]);
			if (labels[i].length() > 2){
				clabelId = m_clabelAlphabet.from_string(labels[i].substr(0, 2) + "seg");
			}
			else{
				clabelId = m_clabelAlphabet.from_string("o-seg");
			}
			m_label_f2c[labelId] = clabelId;

			string curword = normalize_to_lowerwithdigit(words[i]);
			m_word_stats[curword]++;
			for (int j = 0; j < charfeatures[i].size(); j++)
				m_char_stats[charfeatures[i][j]]++;
			for (int j = 0; j < typefeatures[i].size(); j++)
				m_type_stats[j][typefeatures[i][j]]++;
			for (int j = 0; j < sparsefeatures[i].size(); j++)
				m_feat_stats[sparsefeatures[i][j]]++;
		}

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	cout << numInstance << " " << endl;
	cout << "Label num: " << m_labelAlphabet.size() << endl;
	cout << "coarse Label num: " << m_clabelAlphabet.size() << endl;
	cout << "Seg Label num: " << m_seglabelAlphabet.size() << endl;
	cout << "coarse Seg Label num: " << m_cseglabelAlphabet.size() << endl;
	m_labelAlphabet.set_fixed_flag(true);
	m_clabelAlphabet.set_fixed_flag(true);
	m_seglabelAlphabet.set_fixed_flag(true);
	m_cseglabelAlphabet.set_fixed_flag(true);
	ignoreLabels.insert(unknownkey);

	return 0;
}

int Tagger::addTestAlpha(const vector<Instance>& vecInsts) {
	cout << "Adding word Alphabet..." << endl;


	for (int numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->words;
		const vector<vector<string> > &charfeatures = pInstance->charfeatures;
		const vector<vector<string> > &typefeatures = pInstance->typefeatures;
		for (int iter_type = 0; iter_type < typefeatures.size(); iter_type++) {
			assert(m_type_stats.size() == typefeatures[iter_type].size());
		}
		int curInstSize = words.size();
		for (int i = 0; i < curInstSize; ++i) {
			string curword = normalize_to_lowerwithdigit(words[i]);
			if (!m_options.wordEmbFineTune)m_word_stats[curword]++;
			if (!m_options.charEmbFineTune){
				for (int j = 1; j < charfeatures[i].size(); j++){
					m_char_stats[charfeatures[i][j]]++;
				}
			}
			if (!m_options.typeEmbFineTune){
				for (int j = 0; j < typefeatures[i].size(); j++){
					m_type_stats[j][typefeatures[i][j]]++;
				}
			}
		}

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}


	return 0;
}


void Tagger::extractFeature(Feature& feat, const Instance* pInstance, int idx) {
	feat.clear();

	const vector<string>& words = pInstance->words;
	int sentsize = words.size();
	string curWord = idx >= 0 && idx < sentsize ? normalize_to_lowerwithdigit(words[idx]) : nullkey;

	// word features
	feat.words.push_back(curWord);

	// char features

	const vector<vector<string> > &charfeatures = pInstance->charfeatures;

	const vector<string>& cur_chars = charfeatures[idx];
	for (int i = 0; i < cur_chars.size(); i++) {
		feat.chars.push_back(cur_chars[i]);
	}

	const vector<vector<string> > &typefeatures = pInstance->typefeatures;

	const vector<string>& cur_types = typefeatures[idx];
	for (int i = 0; i < cur_types.size(); i++) {
		feat.types.push_back(cur_types[i]);
	}

	const vector<string>& linear_features = pInstance->sparsefeatures[idx];
	for (int i = 0; i < linear_features.size(); i++) {
		feat.linear_features.push_back(linear_features[i]);
	}

}

void Tagger::convert2Example(const Instance* pInstance, Example& exam, bool bTrain) {
	exam.clear();
	const vector<string> &labels = pInstance->labels;
	int curInstSize = labels.size();
	
	for (int i = 0; i < curInstSize; ++i) {
		string orcale = labels[i];

		int numLabel = m_labelAlphabet.size();
		vector<dtype> curlabels;
		for (int j = 0; j < numLabel; ++j) {
			string str = m_labelAlphabet.from_id(j);
			if (str.compare(orcale) == 0)
				curlabels.push_back(1.0);
			else
				curlabels.push_back(0.0);
		}

		exam.m_labels.push_back(curlabels);
		Feature feat;
		extractFeature(feat, pInstance, i);
		exam.m_features.push_back(feat);
	}

	resizeVec(exam.m_seglabels, curInstSize, m_options.maxsegLen, m_seglabelAlphabet.size());
	assignVec(exam.m_seglabels, 0.0);
	vector<segIndex> segs;
	getSegs(labels, segs);
	static int startIndex, disIndex, orcaleId, corcaleId;
	for (int idx = 0; idx < segs.size(); idx++){
		orcaleId =  m_seglabelAlphabet.from_string(segs[idx].label);
		corcaleId = m_seg_f2c[orcaleId];
		startIndex = segs[idx].start;
		disIndex = segs[idx].end - segs[idx].start;
		if (disIndex < m_options.maxsegLen && orcaleId >= 0) { 
			exam.m_seglabels[startIndex][disIndex][orcaleId] = 1.0;
			if (maxLabelLength[orcaleId] < disIndex + 1) maxLabelLength[orcaleId] = disIndex + 1;
			if (maxcLabelLength[corcaleId] < disIndex + 1) maxcLabelLength[corcaleId] = disIndex + 1;
		}
	}

	// O or o
	for (int i = 0; i < curInstSize; ++i) {
		if (labels[i].length() == 1){
			orcaleId = m_seglabelAlphabet.from_string(labels[i]);
			exam.m_seglabels[i][0][orcaleId] = 1.0;
			if (maxLabelLength[orcaleId] < 1) maxLabelLength[orcaleId] = 1;
		}
	}
}

void Tagger::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams, bool bTrain) {
	int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];
		Example curExam;
		convert2Example(pInstance, curExam, bTrain);
		vecExams.push_back(curExam);

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	cout << numInstance << " " << endl;
}

void Tagger::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);
	m_options.showOptions();
	vector<Instance> trainInsts, devInsts, testInsts;
	static vector<Instance> decodeInstResults;
	static Instance curDecodeInst;
	bool bCurIterBetter = false;

	m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
	if (devFile != "")
		m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
	if (testFile != "")
		m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

	//Ensure that each file in m_options.testFiles exists!
	vector<vector<Instance> > otherInsts(m_options.testFiles.size());
	for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
		m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
	}

	//std::cout << "Training example number: " << trainInsts.size() << std::endl;
	//std::cout << "Dev example number: " << trainInsts.size() << std::endl;
	//std::cout << "Test example number: " << trainInsts.size() << std::endl;

	createAlphabet(trainInsts);
	addTestAlpha(devInsts);
	addTestAlpha(testInsts);
	for (int idx = 0; idx < otherInsts.size(); idx++) {
		addTestAlpha(otherInsts[idx]);
	}

	vector<Example> trainExamples, devExamples, testExamples;
	maxLabelLength.resize(m_seglabelAlphabet.size());
	assignVec(maxLabelLength, 0);
	maxcLabelLength.resize(m_cseglabelAlphabet.size());
	assignVec(maxcLabelLength, 0);
	initialExamples(trainInsts, trainExamples, true);
	//print length information
	std::cout << "Predefined max seg length: " << m_options.maxsegLen << std::endl;
	for (int j = 0; j < m_seglabelAlphabet.size(); j++){
		std::cout << "max length of label " << m_seglabelAlphabet.from_id(j) << ": " << maxLabelLength[j] << std::endl;
	}
	m_classifier._loss.initial(maxLabelLength, maxcLabelLength, m_seg_f2c, m_options.maxsegLen, m_options.tuneLambda, 10000);
	initialExamples(devInsts, devExamples);
	initialExamples(testInsts, testExamples);

	vector<int> otherInstNums(otherInsts.size());
	vector<vector<Example> > otherExamples(otherInsts.size());
	for (int idx = 0; idx < otherInsts.size(); idx++) {
		initialExamples(otherInsts[idx], otherExamples[idx]);
		otherInstNums[idx] = otherExamples[idx].size();
	}

	m_word_stats[unknownkey] = m_options.wordCutOff + 1;
	if (m_options.wordFile != "") {
		m_classifier._words.initial(m_word_stats, m_options.wordCutOff, m_options.wordFile, m_options.wordEmbFineTune);
	}
	else{
		m_classifier._words.initial(m_word_stats, m_options.wordCutOff, m_options.wordEmbSize, 0, m_options.wordEmbFineTune);
	}

	int typeNum = m_type_stats.size();
	m_classifier._types.resize(typeNum);
	for (int idx = 0; idx < typeNum; idx++){
		m_type_stats[idx][unknownkey] = 1; // use the s
		if (m_options.typeFiles.size() > idx && m_options.typeFiles[idx] != "") {
			m_classifier._types[idx].initial(m_type_stats[idx], 0, m_options.typeFiles[idx], m_options.typeEmbFineTune);
		}
		else{
			m_classifier._types[idx].initial(m_type_stats[idx], 0, m_options.typeEmbSize, (idx + 1) * 1000, m_options.typeEmbFineTune);
		}
	}

	// use rnnHiddenSize to replace segHiddensize
	m_classifier.setDropValue(m_options.dropProb);
	m_classifier.init(m_options.wordcontext, m_options.charcontext, m_options.hiddenSize, m_options.rnnHiddenSize, m_options.hiddenSize, m_options.segHiddenSize);
	
	m_classifier.setUpdateParameters(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);


	dtype bestDIS = 0;

	int inputSize = trainExamples.size();

	int batchBlock = inputSize / m_options.batchSize;
	if (inputSize % m_options.batchSize != 0)
		batchBlock++;

	srand(0);
	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);

	static Metric eval, metric_dev, metric_test, metric_dev2, metric_test2;
	static vector<Example> subExamples;
	int devNum = devExamples.size(), testNum = testExamples.size();
	for (int iter = 0; iter < m_options.maxIter; ++iter) {
		std::cout << "##### Iteration " << iter << std::endl;

		random_shuffle(indexes.begin(), indexes.end());
		eval.reset();
		for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
			subExamples.clear();
			int start_pos = updateIter * m_options.batchSize;
			int end_pos = (updateIter + 1) * m_options.batchSize;
			if (end_pos > inputSize)
				end_pos = inputSize;

			for (int idy = start_pos; idy < end_pos; idy++) {
				subExamples.push_back(trainExamples[indexes[idy]]);
			}

			int curUpdateIter = iter * batchBlock + updateIter;
			dtype cost = m_classifier.train(subExamples, curUpdateIter);

			eval.overall_label_count += m_classifier._eval.overall_label_count;
			eval.correct_label_count += m_classifier._eval.correct_label_count;

			if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
				//m_classifier.checkgrad(subExamples, curUpdateIter + 1);
				std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
				std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
			}
			m_classifier.updateModel();

		}

		if (devNum > 0) {
			bCurIterBetter = false;
			if (!m_options.outBest.empty())
				decodeInstResults.clear();
			metric_dev.reset();
			metric_dev2.reset();
			for (int idx = 0; idx < devExamples.size(); idx++) {
				vector<string> result_labels;
				predict(devExamples[idx].m_features, result_labels);

				if (m_options.seg){
					devInsts[idx].SegEvaluate(result_labels, metric_dev);
					devInsts[idx].SegUnlabelEvaluate(result_labels, metric_dev2);
				}
				else
					devInsts[idx].Evaluate(result_labels, metric_dev);

				if (!m_options.outBest.empty()) {
					curDecodeInst.copyValuesFrom(devInsts[idx]);
					curDecodeInst.assignLabel(result_labels);
					decodeInstResults.push_back(curDecodeInst);
				}
			}
			std::cout << "dev:" << std::endl;
			metric_dev.print();
			metric_dev2.print();

			if (!m_options.outBest.empty() && metric_dev.getAccuracy() > bestDIS) {
				m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
				bCurIterBetter = true;
			}

			if (testNum > 0) {
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				metric_test2.reset();
				for (int idx = 0; idx < testExamples.size(); idx++) {
					vector<string> result_labels;
					predict(testExamples[idx].m_features, result_labels);

					if (m_options.seg){
						testInsts[idx].SegEvaluate(result_labels, metric_test);
						testInsts[idx].SegUnlabelEvaluate(result_labels, metric_test2);
					}
					else
						testInsts[idx].Evaluate(result_labels, metric_test);

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(testInsts[idx]);
						curDecodeInst.assignLabel(result_labels);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();
				metric_test2.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
				}
			}

			for (int idx = 0; idx < otherExamples.size(); idx++) {
				std::cout << "processing " << m_options.testFiles[idx] << std::endl;
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				metric_test2.reset();
				for (int idy = 0; idy < otherExamples[idx].size(); idy++) {
					vector<string> result_labels;
					predict(otherExamples[idx][idy].m_features, result_labels);

					if (m_options.seg){
						otherInsts[idx][idy].SegEvaluate(result_labels, metric_test);
						otherInsts[idx][idy].SegUnlabelEvaluate(result_labels, metric_test2);
					}
					else
						otherInsts[idx][idy].Evaluate(result_labels, metric_test);

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(otherInsts[idx][idy]);
						curDecodeInst.assignLabel(result_labels);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();
				metric_test2.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
				}
			}

			if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestDIS) {
				std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
				bestDIS = metric_dev.getAccuracy();
				writeModelFile(modelFile);
			}

		}
		// Clear gradients
	}
}

int Tagger::predict(const vector<Feature>& features, vector<string>& outputs) {
	//assert(features.size() == words.size());
	NRMat<int> labelIdx;
	m_classifier.predict(features, labelIdx);
	int seq_size = features.size();
	outputs.resize(seq_size);
	for (int idx = 0; idx < seq_size; idx++) {
		outputs[idx] = nullkey;
	}
	for (int idx = 0; idx < seq_size; idx++) {
		for (int dist = 0; idx + dist < seq_size && dist < m_options.maxsegLen; dist++) {
			if (labelIdx[idx][dist] < 0) continue;
			string label = m_seglabelAlphabet.from_id(labelIdx[idx][dist], unknownkey);
			for (int i = idx; i <= idx + dist; i++){
				if (outputs[i] != nullkey) {
					std::cout << "predict error" << std::endl;
				}
			}
			if (ignoreLabels.find(label) != ignoreLabels.end()){
				for (int i = idx; i <= idx + dist; i++){
					outputs[i] = label;
				}
			}
			else{
				if (dist == 0){
					outputs[idx] = "s-" + label;
				}
				else{
					outputs[idx] = "b-" + label;
					for (int i = idx + 1; i < idx + dist; i++){
						outputs[i] = "m-" + label;
					}
					outputs[idx + dist] = "e-" + label;
				}
			}			
		}
	}

	for (int idx = 0; idx < seq_size; idx++) {
		if (outputs[idx] == nullkey){
			std::cout << "predict error" << std::endl;
		}
	}

	return 0;
}

void Tagger::test(const string& testFile, const string& outputFile, const string& modelFile) {
	loadModelFile(modelFile);
	vector<Instance> testInsts;
	m_pipe.readInstances(testFile, testInsts);

	vector<Example> testExamples;
	initialExamples(testInsts, testExamples);

	int testNum = testExamples.size();
	vector<Instance> testInstResults;
	Metric metric_test, metric_test2;
	metric_test.reset();
	metric_test2.reset();
	for (int idx = 0; idx < testExamples.size(); idx++) {
		vector<string> result_labels;
		predict(testExamples[idx].m_features, result_labels);
		if (m_options.seg){
			testInsts[idx].SegEvaluate(result_labels, metric_test);
			testInsts[idx].SegUnlabelEvaluate(result_labels, metric_test2);
		}
		else
			testInsts[idx].Evaluate(result_labels, metric_test);		
		Instance curResultInst;
		curResultInst.copyValuesFrom(testInsts[idx]);
		curResultInst.assignLabel(result_labels);
		testInstResults.push_back(curResultInst);
	}
	std::cout << "test:" << std::endl;
	metric_test.print();
	metric_test2.print();

	m_pipe.outputAllInstances(outputFile, testInstResults);

}


void Tagger::loadModelFile(const string& inputModelFile) {

}

void Tagger::writeModelFile(const string& outputModelFile) {

}

int main(int argc, char* argv[]) {

	std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	dsr::Argument_helper ah;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

	ah.process(argc, argv);

	Tagger segmentor;
	segmentor.m_pipe.max_sentense_size = ComputionGraph::max_sentence_length;
	if (bTrain) {
		segmentor.train(trainFile, devFile, testFile, modelFile, optionFile);
	}
	else {
		segmentor.test(testFile, outputFile, modelFile);
	}

	//test(argv);
	//ah.write_values(std::cout);
}
