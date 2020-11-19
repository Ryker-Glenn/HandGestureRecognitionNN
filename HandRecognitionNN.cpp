// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include "NN.h"

using namespace std;

void genData(string filename) {
	ofstream file1(filename + "-in");
	ofstream file2(filename + "-out");
	for (int r = 0; r < 1000; r++) {
		Scalar x = rand() / Scalar(RAND_MAX);
		Scalar y = rand() / Scalar(RAND_MAX);
		file1 << x << ", " << y << std::endl;
		file2 << 2 * x + 10 + y << std::endl;
	}
	file1.close();
	file2.close();
}

void ReadCSV(string filename, vector<RowVector*>& data) {
	data.clear();
	ifstream file(filename);
	string line, word;
	// determine number of columns in file 
	getline(file, line, '\n');
	stringstream ss(line);
	vector<Scalar> parsed_vec;
	while (getline(ss, word, ',')) {
		parsed_vec.push_back(Scalar(std::stof(&word[0])));
	}
	unsigned int cols = parsed_vec.size();
	data.push_back(new RowVector(cols));
	for (unsigned int i = 0; i < cols; i++) {
		data.back()->coeffRef(1, i) = parsed_vec[i];
	}

	// read the file 
	if (file.is_open()) {
		while (getline(file, line, '\n')) {
			stringstream ss(line);
			data.push_back(new RowVector(1, cols));
			unsigned int i = 0;
			while (getline(ss, word, ',')) {
				data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
				i++;
			}
		}
	}
}

typedef vector<RowVector*> dat;
int main() {
	NeuralNetwork n({ 2, 3, 1 });
	dat in_dat, out_dat;
	genData("test");
	ReadCSV("test-in", in_dat);
	ReadCSV("test-out", out_dat);
	n.train(in_dat, out_dat);
	return 0;
}