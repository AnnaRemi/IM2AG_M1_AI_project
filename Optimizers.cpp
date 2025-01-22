#include "../include/Optimizers.h"
#include "../include/Layer.h"
#include <memory>
#include <random>

GradientDescent::GradientDescent(double lr){
    learning_rate = lr;
}

void GradientDescentWithMomentum::update_params(std::shared_ptr<Layer> layer) {
    std::vector<std::vector<double>> weights = layer->getWeights();
    std::vector<std::vector<double>> dweights = layer->getDweights();
    std::vector<std::vector<double>> biases = layer->getBiases();
    std::vector<std::vector<double>> dbiases = layer->getDbiases();

    if (velocity_weights.empty()) {
        velocity_weights = std::vector<std::vector<double>>(weights.size(), std::vector<double>(weights[0].size(), 0.0));
    }

    if (velocity_biases.empty()) {
        velocity_biases = std::vector<std::vector<double>>(biases.size(), std::vector<double>(biases[0].size(), 0.0));
    }

    for (int i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < weights[0].size(); ++j) {
            velocity_weights[i][j] = momentum * velocity_weights[i][j] - learning_rate * dweights[i][j];
            weights[i][j] += velocity_weights[i][j];
        }
    }

    for (int i = 0; i < biases.size(); ++i) {
        for (int j = 0; j < biases[0].size(); ++j) {
            velocity_biases[i][j] = momentum * velocity_biases[i][j] - learning_rate * dbiases[i][j];
            biases[i][j] += velocity_biases[i][j];
        }
    }

    layer->setWeights(weights);
    layer->setBiases(biases);
}

void GradientDescent::update_params(std::shared_ptr<Layer> layer){
    std::vector<std::vector<double>> weights = layer->getWeights();
    std::vector<std::vector<double>> dweights = layer->getDweights();
    std::vector<std::vector<double>> biases = layer->getBiases();
    std::vector<std::vector<double>> dbiases = layer->getDbiases();


    
    /*std::cout << "dweights" << std::endl;
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

    
    for (int i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < weights[0].size(); ++j) { 
            weights[i][j] += (- learning_rate * dweights[i][j]);
        }
    }
    
    
    for (int i = 0; i < biases.size(); ++i) {
        for (int j = 0; j < biases[0].size(); ++j) { 
            biases[i][j] += (- learning_rate * dbiases[i][j]);
        }
    }
    
    /*std::cout << "dbiases" << std::endl;
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

    
    layer->setWeights(weights);
    layer->setBiases(biases);
}


/**
 * @brief Generates a random matrix with values between minVal and maxVal.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param minVal Minimum value for the random elements.
 * @param maxVal Maximum value for the random elements.
 * @return A matrix of dimensions rows x cols filled with random values.
 */
std::vector<std::vector<double>> RandomVec(int rows, int cols, double minVal = -1.0, double maxVal = 1.0)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(minVal, maxVal);
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = dist(gen);
    return matrix;
}

void RandomUpdate::update_params(std::shared_ptr<Layer> layer){
    //std::cout << "In update " << layer->getWeights().size() << std::endl;
    // Random weight and bias updates
    layer->setWeights(RandomVec(layer->getWeights().size(), layer->getWeights()[0].size()));
    layer->setBiases(RandomVec(layer->getBiases().size(), layer->getBiases()[0].size()));
}