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
    //std::cout << "Creating biases " << std::endl;
    biases.resize( 1 );
    for(int i = 0; i< 1; i++){
        biases[i].resize( numNeurons );
        for(int j = 0; j< numNeurons; j++){
            biases[i][j] = 0.0;
        }
    }
    //std::cout <<  " Cols: " << numNeurons << std::endl;
}

Layer::Layer( int ni, int nn){
    numInputs = ni;
    numNeurons = nn;
    //std::cout << "Creating biases " << std::endl;
    biases.resize( 1 );
    for(int i = 0; i< 1; i++){
        biases[i].resize( numNeurons );
        for(int j = 0; j< numNeurons; j++){
            biases[i][j] = 0.0;
        }
    }
    //std::cout <<  " Cols: " << numNeurons << std::endl;
}

void Layer::setWeights(std::vector<std::vector<double>> w){
    weights = w;
}

void Layer::setBiases(std::vector<std::vector<double>> b){
    biases = b;
}

//Getters
std::vector<std::vector<double>> Layer::getWeights() const{return weights;}

std::vector<std::vector<double>> Layer::getBiases() const{return biases;}

std::vector<std::vector<double>> Layer::getDweights() const{return dweights;}

std::vector<std::vector<double>> Layer::getDbiases() const{return dbiases;}


//std::shared_ptr<Matrix> Layer::getOutput() const{return output;}
std::vector<std::vector<double>> Layer::getOutput() const{return output;}
std::vector<std::vector<double>> Layer::getDinputs() const{return dinputs;}

/*void Layer::multiply( std::shared_ptr<Matrix> mat_weights,  std::shared_ptr<Matrix> mat_inputs){
    if (mat_inputs->getNumCols() != mat_weights->getNumRows())
    {
        throw std::invalid_argument("Invalid dimensions. Remember to use the transpose of the mat_weights matrix");
    }

    int r = mat_inputs->getNumRows();
    int c = mat_weights->getNumRows();

    
    output = std::make_shared<Matrix>(mat_inputs->getNumRows(),mat_weights->getNumCols());

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            double sum = 0;
            for (int k = 0; k < c; ++k) {
                sum += mat_inputs->getValue(i,k) * mat_weights->getValue(k,j);
            }
            output->setValue(i,j, sum);
        }
    }
}*/

void Layer::add(std::shared_ptr<Matrix> mat_outputs){
    //std::cout <<  "output->getNumCols() " << mat_outputs->getNumRows() << " mat_outputs->getNumRows() " << mat_outputs->getNumCols() << std::endl;
    for (int i = 0; i < mat_outputs->getNumRows(); ++i) {
        for (int j = 0; j < mat_outputs->getNumCols(); ++j) {
            mat_outputs->setValue(i,j, mat_outputs->getValue(i,j) + biases[0][j]);
        }
    }
    //std::cout <<  "mat_outputs->getNumCols() " << mat_outputs->getNumRows() << " mat_outputs->getNumRows() " << mat_outputs->getNumCols() << std::endl;
}



void Layer::forward(std::vector<std::vector<double>> i){
    std::cout <<  "Layer inputs: (" << i.size() << ", " << i[0].size() << ")" << std::endl ;
    
    std::shared_ptr<Matrix> mat_inputs = std::make_shared<Matrix>(i.size(), i[0].size(), i);
    inputs = i;
    
    //std::cout <<  "Inputs " << numInputs << " Neurons " << numNeurons << std::endl ;
    //mat_inputs->print();
    std::shared_ptr<Matrix> mat_weights = std::make_shared<Matrix>(numInputs, numNeurons);
    
    //std::cout <<  "Layer weights: (" << mat_weights->getNumRows() << ", " << mat_weights->getNumCols() << ")" << std::endl ;

    weights = mat_weights->getValues();
    //std::cout <<  "Layer weights: (" << weights.size() << ", " << weights[0].size() << ")" << std::endl ;
    //std::cout <<  "Weights" << std::endl ;
    //mat_weights->print();
    //mat_weights = mat_weights->transpose();
    //std::cout <<  "Weights transpose" << std::endl ;
    //mat_weights->print();

    std::shared_ptr<Matrix> mat_outputs = mat_inputs->dotProduct(*mat_weights);
    //multiply(mat_weights, mat_inputs);
    

    
    add(mat_outputs);

    output = mat_outputs->getValues();
    //std::cout << "Finished forward layer" << std::endl;
}


void Layer::backward(std::vector<std::vector<double>> dvalues){
    
    std::shared_ptr<Matrix> mat_dvalues = std::make_shared<Matrix>(dvalues.size(), dvalues[0].size(), dvalues);
    //std::cout << "mat_dvalues backwards" << std::endl;
    //mat_dvalues->print();
    //std::cout << "before mat_inputs backwards" << std::endl;
    std::shared_ptr<Matrix> mat_inputs = std::make_shared<Matrix>(inputs.size(), inputs[0].size(), inputs);
    //std::cout << "mat_inputs backwards" << std::endl;
    //mat_inputs->print();
    std::shared_ptr<Matrix> mat_inputs_T = mat_inputs->transpose();
   
    std::shared_ptr<Matrix> mat_dweights = mat_inputs_T->dotProduct(*mat_dvalues);
   
    dweights = mat_dweights->getValues();
    
    //std::cout << "mat_dweights backwards" << std::endl;
   // mat_dweights->print();

    dbiases = mat_dvalues->sumOverRows();
   

    std::shared_ptr<Matrix> mat_weights = std::make_shared<Matrix>(weights.size(), weights[0].size(), weights);
    std::shared_ptr<Matrix> mat_weights_T = mat_weights->transpose();
    std::shared_ptr<Matrix> mat_dinputs =  mat_dvalues->dotProduct(*mat_weights_T);
   
   
    dinputs = mat_dinputs->getValues();
    /*std::cout <<  " dinputs " <<  std::endl;
    for (int j = 0; j < dinputs[0].size(); j++) { 
        std::cout << dinputs[0][j] << " " ; 
    }
    std::cout << std::endl ;*/
}

/*std::vector<std::vector<double>> Layer::backward(std::vector<std::vector<double>> dvalues){
    Matrix mat_dvalues = Matrix(dvalues.size(), dvalues[0].size(), dvalues);
    Matrix mat_inputs = Matrix(inputs.size(), inputs[0].size(), inputs);
    dweights = (&mat_inputs)->transpose().dotProduct(mat_dvalues)->getValues();
    dbiases = mat_dvalues.sumOverRows();

    // std::cout << "columns m1 : " << mat_dvalues.getNumCols() << std::endl;
    // std::cout << "rows m1 : " << mat_dvalues.getNumRows() << std::endl;
    // std::cout << "rows m2 : " << weights->transpose().getNumRows() << std::endl;
    // std::cout << "columns m2 : " << weights->transpose().getNumCols() << std::endl;
    Matrix mat_weights = Matrix(weights.size(), weights[0].size(), weights);
    dinputs = mat_dvalues.dotProduct((&mat_weights)->transpose()->getValues();
}*/





