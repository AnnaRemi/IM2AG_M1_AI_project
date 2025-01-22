#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>
#include <random>

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

    //For Adam
    weight_momentums.resize( w.size() );
    for(int i = 0; i< w.size(); i++){
        weight_momentums[i].resize( w[0].size() );
        for(int j = 0; j< w[0].size(); j++){
            weight_momentums[i][j] = 0.0;
        }
    };   
    weight_cache.resize( w.size() );
    for(int i = 0; i< w.size(); i++){
        weight_cache[i].resize( w[0].size() );
        for(int j = 0; j< w[0].size(); j++){
            weight_cache[i][j] = 0.0;
        }
    };
    bias_momentums.resize( 1 );
    for(int i = 0; i< 1; i++){
        bias_momentums[i].resize( numNeurons );
        for(int j = 0; j< numNeurons; j++){
            bias_momentums[i][j] = 0.0;
        }
    }; 
    bias_cache.resize( 1 );
    for(int i = 0; i< 1; i++){
        bias_cache[i].resize( numNeurons );
        for(int j = 0; j< numNeurons; j++){
            bias_cache[i][j] = 0.0;
        }
    };
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

     //For Adam
    weight_momentums.resize( weights.size() );
    for(int i = 0; i< weights.size(); i++){
        weight_momentums[i].resize( weights[0].size() );
        for(int j = 0; j< weights[0].size(); j++){
            weight_momentums[i][j] = 0.0;
        }
    };   
    weight_cache.resize( weights.size() );
    for(int i = 0; i< weights.size(); i++){
        weight_cache[i].resize( weights[0].size() );
        for(int j = 0; j< weights[0].size(); j++){
            weight_cache[i][j] = 0.0;
        }
    };
    bias_momentums.resize( 1 );
    for(int i = 0; i< 1; i++){
        bias_momentums[i].resize( numNeurons );
        for(int j = 0; j< numNeurons; j++){
            bias_momentums[i][j] = 0.0;
        }
    }; 
    bias_cache.resize( 1 );
    for(int i = 0; i< 1; i++){
        bias_cache[i].resize( numNeurons );
        for(int j = 0; j< numNeurons; j++){
            bias_cache[i][j] = 0.0;
        }
    };
}

void Layer::setWeights(const std::vector<std::vector<double>>& w){
    weights = w;
}


void Layer::setBiases(const std::vector<std::vector<double>>& b){
    biases = b;
}


void Layer::setWeightMomentums(const std::vector<std::vector<double>>& wm) {
    weight_momentums = wm;
}

void Layer::setBiasMomentums(const std::vector<std::vector<double>>& bm) {
    bias_momentums = bm;
}

void Layer::setWeightCache(const std::vector<std::vector<double>>& wc) {
    weight_cache = wc;
}

void Layer::setBiasCache(const std::vector<std::vector<double>>& bc) {
    bias_cache = bc;
}

//Getters
std::vector<std::vector<double>> Layer::getWeights() const{return weights;}
std::vector<std::vector<double>> Layer::getBiases() const{return biases;}
std::vector<std::vector<double>> Layer::getDweights() const{return dweights;}
std::vector<std::vector<double>> Layer::getDbiases() const{return dbiases;}
std::vector<std::vector<double>> Layer::getOutput() const{return output;}
std::vector<std::vector<double>> Layer::getDinputs() const{return dinputs;}
std::vector<std::vector<double>> Layer::getWeightMomentums() const{return weight_momentums;}
std::vector<std::vector<double>> Layer::getBiasMomentums() const{return bias_momentums;}
std::vector<std::vector<double>> Layer::getWeightCache() const{return weight_cache;}
std::vector<std::vector<double>> Layer::getBiasCache() const{return bias_cache;}

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
    /*std::cout << "biases" << std::endl;
    std::cout <<  "[ " ;
    for (int i = 0; i < biases.size(); i++) {
        std::cout <<  "[ " ;
        for (int j = 0; j < biases[0].size(); j++) { 
            std::cout << biases[i][j] << " " ; 
        }
        std::cout <<  " ]" << std::endl ;
    }
    
    std::cout <<  " ]" << std::endl ;
    std::cout <<  " end" << std::endl ;*/
}



void Layer::forward(std::vector<std::vector<double>> i){
    std::shared_ptr<Matrix> mat_inputs = std::make_shared<Matrix>(i.size(), i[0].size(), i);
    inputs = i;
    
    /*std::cout << "mat_inputs" << std::endl;
    std::cout <<  "[ " ;
    for (int i = 0; i < 5; i++) {
        std::cout <<  "[ " ;
        for (int j = 0; j < mat_inputs->getNumCols(); j++) { 
            std::cout << mat_inputs->getValue(i,j) << " " ; 
        }
        std::cout <<  " ]" << std::endl ;
    }
    
    std::cout <<  " ]" << std::endl ;
    std::cout <<  " end" << std::endl ;*/
    
    std::shared_ptr<Matrix> mat_weights = std::make_shared<Matrix>(numInputs, numNeurons, weights);
    
    /*std::cout << "mat_weights" << std::endl;
    std::cout <<  "[ " ;
    for (int i = 0; i < mat_weights->getNumRows(); i++) {
        std::cout <<  "[ " ;
        for (int j = 0; j < mat_weights->getNumCols(); j++) { 
            std::cout << mat_weights->getValue(i,j) << " " ; 
        }
        std::cout <<  " ]" << std::endl ;
    }
    
    std::cout <<  " ]" << std::endl ;
    std::cout <<  " end" << std::endl ;*/

    
    std::shared_ptr<Matrix> mat_outputs = mat_inputs->dotProduct(*mat_weights);
    

    add(mat_outputs);

    
    output = mat_outputs->getValues();
}


void Layer::backward(std::vector<std::vector<double>> dvalues){
    
    std::shared_ptr<Matrix> mat_dvalues = std::make_shared<Matrix>(dvalues.size(), dvalues[0].size(), dvalues);

    /*std::cout << "mat_dvalues" << std::endl;
    std::cout <<  "[ " ;
    for (int i = 0; i < 5; i++) {
        std::cout <<  "[ " ;
        for (int j = 0; j < mat_dvalues->getNumCols(); j++) { 
            std::cout << mat_dvalues->getValue(i,j) << " " ; 
        }
        std::cout <<  " ]" << std::endl ;
    }
    
    std::cout <<  " ]" << std::endl ;
    std::cout <<  " end" << std::endl ;*/

    std::shared_ptr<Matrix> mat_inputs = std::make_shared<Matrix>(inputs.size(), inputs[0].size(), inputs);

    std::shared_ptr<Matrix> mat_inputs_T = mat_inputs->transpose();
    
    /*std::cout << "mat_inputs" << mat_inputs->getNumRows() << " " << mat_inputs->getNumCols() << std::endl;
    std::cout <<  "[ " ;
    for (int i = 0; i < 5; i++) {
        std::cout <<  "[ " ;
        for (int j = 0; j < mat_inputs->getNumCols(); j++) { 
            std::cout << mat_inputs->getValue(i,j) << " " ; 
        }
        std::cout <<  " ]" << std::endl ;
    }
    
    std::cout <<  " ]" << std::endl ;
    std::cout <<  " end" << std::endl ;*/
   
    std::shared_ptr<Matrix> mat_dweights = mat_inputs_T->dotProduct(*mat_dvalues);
    
   
    dweights = mat_dweights->getValues();
    
    /*std::cout << "dweights layer backward" << std::endl;
    std::cout <<  "[ " ;
    for (int i = 0; i < dweights.size(); i++) {
        std::cout <<  "[ " ;
        for (int j = 0; j < dweights[0].size(); j++) { 
            std::cout << dweights[i][j] << " " ; 
        }
        std::cout <<  " ]" << std::endl ;
    }
    
    std::cout <<  " ]" << std::endl ;
    std::cout <<  " end" << std::endl ;*/

    dbiases = mat_dvalues->sumOverRows();
    
    /*std::cout << "dbiases layer backward" << std::endl;
    std::cout <<  "[ " ;
    for (int i = 0; i < dbiases.size(); i++) {
        std::cout <<  "[ " ;
        for (int j = 0; j < dbiases[0].size(); j++) { 
            std::cout << dbiases[i][j] << " " ; 
        }
        std::cout <<  " ]" << std::endl ;
    }
    
    std::cout <<  " ]" << std::endl ;
    std::cout <<  " end" << std::endl ;*/
   

    std::shared_ptr<Matrix> mat_weights = std::make_shared<Matrix>(weights.size(), weights[0].size(), weights);
    std::shared_ptr<Matrix> mat_weights_T = mat_weights->transpose();
    std::shared_ptr<Matrix> mat_dinputs =  mat_dvalues->dotProduct(*mat_weights_T);
   
   
    dinputs = mat_dinputs->getValues();
   
}







