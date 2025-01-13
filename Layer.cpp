#include <iostream>
#include <cstdlib>
#include <vector>

#include "Layer.h"
#include "Matrix.h"

Layer::Layer(int ni, int nn){
    numInputs = ni;
    numNeurons = nn;
    weights = Matrix(ni, nn);
    biases = Matrix(1, nn);
}

Layer::Layer(int ni, int nn, std::vector<std::vector<double>> w, std::vector<std::vector<double>> b){
    if(ni!=int(w.size()) || nn!=int(w[0].size())){
        throw std::invalid_argument("Error in the construction of the Layer : the given number of inputs or of neurons does not correspond to the dimensions of the given weight matrix");
    }
    numInputs = ni;
    numNeurons = nn;
    weights = Matrix(w.size(),w[0].size(),w);
    biases = Matrix(b.size(), b[0].size(), b);
}

Layer::Layer(int ni, int nn, std::vector<std::vector<double>> b){
    if (nn != int(b[0].size()))
    {
        throw std::invalid_argument("Error in the construction of the Layer : the given number of neuron does not correspond to the dimensions of the given bias matrix");
    }
    numInputs = ni;
    numNeurons = nn;
    weights = Matrix(ni, nn);
    biases = Matrix(b.size(), b[0].size(), b);
}

Matrix Layer::getOutput() const{return output;}

Matrix Layer::get_dinputs() const { return dinputs; }

Matrix Layer::get_dweights() const { return dweights; }

Matrix Layer::get_dbiases() const { return dbiases; }

Matrix Layer::getWeights() const { return weights; }

Matrix Layer::getBiases() const { return biases; }

Matrix Layer::getDweights() const { return dweights; }

Matrix Layer::getDbiases() const { return dbiases; }
  

/**
 * @brief Performs the multiplication of mat_inputs with mat_weights given as parameters and stores it in output
 * 
 * output = mat_inputs * mat_weights = mat_inputs.dotProduct(mat_weights)
 *
 * @param mat_weights The matrix of weights with the weight of each neuron on each columns
 * @param mat_inputs
 */
void Layer::multiply(){
    //std::cout <<  "inputs.getNumCols() " << inputs.getNumCols() << " weights.getNumRows() " << weights.getNumRows() << std::endl;
    output = inputs.dotProduct(weights);
}

void Layer::add(){
    for (int i = 0; i < numNeurons; ++i)
    {
        for (int j = 0; j < output.getNumCols(); ++j)
        {
            output.setValue(j, i, output.getValue(j, i) + biases.getValue(0, i));
        }
    }
}

void Layer::forward(std::vector<std::vector<double>> i){
    inputs = i;
    //Don't need to take mat_weight_transpose() because the matrix mat_weights the weights is already made such that each column is the weights of each neurons
    multiply();
    add();
}

void Layer::backward(std::vector<std::vector<double>> dvalues){
    
    Matrix mat_dvalues = Matrix(dvalues.size(), dvalues[0].size(), dvalues);
    dweights = Matrix(inputs.getNumCols(), dvalues[0].size());
    dweights = inputs.transpose().dotProduct(mat_dvalues);
    dbiases = mat_dvalues.sumOverRows();
    dinputs = Matrix(mat_dvalues.getNumCols(), weights.getNumCols());
    dinputs = mat_dvalues.dotProduct(weights.transpose());
}

void Layer::print_weights(){
    weights.print();
}


/**
 * Methods and Constructors for class Optimizer
 * 
 */

Optimizer::Optimizer(double l):learning_rate(l){}


void Optimizer::update_parameters(Layer &L){
    L.weights +=  L.dweights * (-learning_rate);
    L.biases += L.dbiases * (-learning_rate);
}

