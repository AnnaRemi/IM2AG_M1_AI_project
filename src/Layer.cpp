#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>

#include "../include/Layer.h"
#include "../include/Matrix.h"


Layer::Layer( int ni, int nn, std::vector<std::vector<double>> w){
    numInputs = ni;
    numNeurons = nn;
    weights = w;
    biases.resize( 1 );
    for(int i = 0; i< 1; i++){
        biases[i].resize( numNeurons );
        for(int j = 0; j< numNeurons; j++){
            biases[i][j] = 0.0;
        }
    }
}

Layer::Layer( int ni, int nn){
    numInputs = ni;
    numNeurons = nn;
    biases.resize( 1 );
    for(int i = 0; i< 1; i++){
        biases[i].resize( numNeurons );
        for(int j = 0; j< numNeurons; j++){
            biases[i][j] = 0.0;
        }
    }
}

void Layer::setWeights(const std::vector<std::vector<double>>& w){
    weights = w;
}

void Layer::setBiases(const std::vector<std::vector<double>>& b){
    biases = b;
}

//Getters
std::vector<std::vector<double>> Layer::getWeights() const{return weights;}
std::vector<std::vector<double>> Layer::getBiases() const{return biases;}
std::vector<std::vector<double>> Layer::getDweights() const{return dweights;}
std::vector<std::vector<double>> Layer::getDbiases() const{return dbiases;}
std::vector<std::vector<double>> Layer::getOutput() const{return output;}
std::vector<std::vector<double>> Layer::getDinputs() const{return dinputs;}

void Layer::printMatrix(std::vector<std::vector<double>> matrix){
    std::cout <<  "[ " ;
    for (int i = 0; i < matrix.size(); i++) {
        std::cout <<  "[ " ;
        for (int j = 0; j < matrix[0].size(); j++) { 
            std::cout << matrix[i][j] << " " ; 
        }
        std::cout <<  " ]" << std::endl ;
    }
    
    std::cout <<  " ]" << std::endl ;
    std::cout <<  " end" << std::endl ;
}

void Layer::add(std::shared_ptr<Matrix> mat_outputs){
    for (int i = 0; i < mat_outputs->getNumRows(); ++i) {
        for (int j = 0; j < mat_outputs->getNumCols(); ++j) {
            mat_outputs->setValue(i,j, mat_outputs->getValue(i,j) + biases[0][j]);
        }
    }
}



void Layer::forward(std::vector<std::vector<double>> i){
    std::shared_ptr<Matrix> mat_inputs = std::make_shared<Matrix>(i.size(), i[0].size(), i);
    inputs = i;
    
    std::shared_ptr<Matrix> mat_weights = std::make_shared<Matrix>(numInputs, numNeurons, weights);

    std::shared_ptr<Matrix> mat_outputs = mat_inputs->dotProduct(*mat_weights);
   
    add(mat_outputs);

    output = mat_outputs->getValues();
}


void Layer::backward(std::vector<std::vector<double>> dvalues){
    
    std::shared_ptr<Matrix> mat_dvalues = std::make_shared<Matrix>(dvalues.size(), dvalues[0].size(), dvalues);

    std::shared_ptr<Matrix> mat_inputs = std::make_shared<Matrix>(inputs.size(), inputs[0].size(), inputs);

    std::shared_ptr<Matrix> mat_inputs_T = mat_inputs->transpose();
   
    std::shared_ptr<Matrix> mat_dweights = mat_inputs_T->dotProduct(*mat_dvalues);
   
    dweights = mat_dweights->getValues();

    dbiases = mat_dvalues->sumOverRows();
   

    std::shared_ptr<Matrix> mat_weights = std::make_shared<Matrix>(weights.size(), weights[0].size(), weights);
    std::shared_ptr<Matrix> mat_weights_T = mat_weights->transpose();
    std::shared_ptr<Matrix> mat_dinputs =  mat_dvalues->dotProduct(*mat_weights_T);
   
   
    dinputs = mat_dinputs->getValues();
   
}







