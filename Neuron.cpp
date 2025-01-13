
#include <iostream>
#include <cstdlib>
#include <vector>
#include "Neuron.h"

// Default constructor
Neuron::Neuron() {}

// Parameterized constructor
Neuron::Neuron(std::vector<std::vector<double>> w, std::vector<std::vector<double>> i, double b){
    weights = w;
    inputs = i;
    bias = b;
}


double Neuron::getBias(){
    return bias;
}

void  Neuron::setWeights(std::vector<std::vector<double>> w){
    weights = w;
}

//v1 is m*n and v2 is n*p
//dotProduct returns m*p matrix
/*double  Neuron::dotProduct(){
    if (weights.size() != inputs.size()) {
        throw std::invalid_argument("Vector dimensions do not match for dot product.");
    }
    double sum = 0;
    for(int i=0; i<weights.size(); i++){
        sum += weights[i]*inputs[i];
    }
    return sum;
}*/

/*template <>
double Neuron<std::pair<std::vector<double>, std::vector<std::vector<double>>>>::dotProduct() {
    const std::vector<double>& vec = weights;
    const std::vector<std::vector<double>>& mat = inputs;

    if (vec.size() != mat.size()) {
        throw std::invalid_argument("Vector and matrix dimensions do not match for multiplication.");
    }

    std::vector<double> result(mat[0].size(), 0.0);

    for (size_t i = 0; i < mat[0].size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += vec[j] * mat[j][i];
        }
    }

    double sum = 0;
    for (double val : result) {
        sum += val;
    }

    return sum;
}*/

/*std::vector<std::vector<double>> Neuron::dotProduct() {
    if (inputs[0].size() != weights.size()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    std::vector<std::vector<double>> result(inputs.size(), weights[0].size());

    for (int i = 0; i < inputs.size(); ++i) {
        for (int j = 0; j < weights[0].size(); ++j) {
            for (int k = 0; k < inputs[0].size(); ++k) {
                result[i][j] += inputs[i][k] * weights[k][j];
            }
        }
    }

    return result;
}


std::vector<std::vector<double>> Neuron::output(){

    std::vector<std::vector<double>> output(inputs.size(), weights[0].size());

    std::vector<std::vector<double>> dot_product = dotProduct();
    for (int i = 0; i < dot_product.size(); ++i) {
        for (int j = 0; j < dot_product[0].size(); ++j) {
            output[i][j] += dot_product[i][j] + bias;
        }
    }
    return output;
}*/