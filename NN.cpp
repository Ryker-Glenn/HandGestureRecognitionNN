#include <iostream>
#include "NN.h"

using namespace std;

Scalar activationFunction(Scalar x) {
	return tanhf(x);
}

Scalar activationFunctionDerivative(Scalar x) {
	return 1 - tanhf(x) * tanhf(x);
}

NeuralNetwork::NeuralNetwork(vector<int> topology, Scalar learningRate) {
	this->topology = topology;
	this->learningRate = learningRate;
	for ( int i = 0; i < topology.size(); i++) {
		// initialze neuron layers 
		if (i == topology.size() - 1)
			neuronLayers.push_back(new RowVector(topology[i]));
		else
			neuronLayers.push_back(new RowVector(topology[i] + 1));

		// initialize cache and delta vectors 
		cacheLayers.push_back(new RowVector(neuronLayers.size()));
		deltas.push_back(new RowVector(neuronLayers.size()));

		// vector.back() gives the handle to recently added element 
		// coeffRef gives the reference of value at that place  
		// (using this as we are using pointers here) 
		if (i != topology.size() - 1) {
			neuronLayers.back()->coeffRef(topology[i]) = 1.0;
			cacheLayers.back()->coeffRef(topology[i]) = 1.0;
		}

		// initialze weights matrix 
		if (i > 0) {
			if (i != topology.size() - 1) {
				weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
				weights.back()->setRandom();
				weights.back()->col(topology[i]).setZero();
				weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
			}
			else {
				weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
				weights.back()->setRandom();
			}
		}
	}
}

void NeuralNetwork::propagateForward(RowVector& input) {
	// set input to input layer
	// block returns a part of the given vector or matrix 
    // block takes 4 arguments : startRow, startCol, blockRows, blockCols 
	neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

	// propagate the data forawrd 
	for ( int i = 1; i < topology.size(); i++) {
		// already explained above 
		(*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
	}

	// apply the activation function to your network 
	// unaryExpr applies the given function to all elements of CURRENT_LAYER 
	for ( int i = 1; i < topology.size() - 1; i++) {
		neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(ptr_fun(activationFunction));
	}
}

void NeuralNetwork::propagateBackward(RowVector& output) {
	calcErrors(output);
	updateWeights();
}

void NeuralNetwork::calcErrors(RowVector& output) {
	// calculate the errors made by neurons of last layer 
	(*deltas.back()) = output - (*neuronLayers.back());

	// error calculation of hidden layers is different 
	// we will begin by the last hidden layer 
	// and we will continue till the first hidden layer 
	for ( int i = topology.size() - 2; i > 0; i--) {
		(*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
	}
}

void NeuralNetwork::updateWeights() {
	// topology.size()-1 = weights.size() 
	for (unsigned int i = 0; i < topology.size() - 1; i++) {
		// in this loop we are iterating over the different layers (from first hidden to output layer) 
		// if this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols 
		// if this layer not the output layer, there is a bias neuron and number of neurons specified = number of cols -1 
		if (i != topology.size() - 2) {
			for (int c = 0; c < weights[i]->cols() - 1; c++) {
				for (int r = 0; r < weights[i]->rows(); r++) {
					weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
				}
			}
		}
		else {
			for ( int c = 0; c < weights[i]->cols(); c++) {
				for ( int r = 0; r < weights[i]->rows(); r++) {
					weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
				}
			}
		}
	}
}

void NeuralNetwork::train(vector<RowVector*> in, vector<RowVector*> out) {
	for (int i = 0; i < in.size(); i++) {
		cout << "Input to neural network is : " << *in[i] << endl;
		propagateForward(*in[i]);
		cout << "Expected output is : " << *out[i] << endl;
		cout << "Output produced is : " << *neuronLayers.back() << std::endl;
		propagateBackward(*out[i]);
		cout << "MSE : " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << endl;
	}
}
