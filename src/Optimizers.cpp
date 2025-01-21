#include "../include/Optimizers.h"
#include "../include/Layer.h"
#include <memory>
#include <random>
#include <cmath>

GradientDescent::GradientDescent(double lr){
    learning_rate = lr;
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


Adam::Adam(double lr, double dec, double ep, double b1, double b2){
    learning_rate = lr;
    current_learning_rate = lr;
    decay= dec;
    iterations = 0;
    epsilon = ep;
    beta_1 = b1;
    beta_2 = b2;
}

double Adam::getCurrent_learning_rate(){
    return current_learning_rate;
}

void Adam::pre_update_params(){
    if(decay){
        current_learning_rate = learning_rate * ( 1 / (1 + decay * iterations));
    }
    
}

void Adam::update_params(std::shared_ptr<Layer> layer){
    std::vector<std::vector<double>> weight_momentums = layer->getWeightMomentums();
    std::vector<std::vector<double>> weight_cache = layer->getWeightCache();
    std::vector<std::vector<double>> bias_momentums = layer->getBiasMomentums();
    std::vector<std::vector<double>> bias_cache = layer->getBiasCache();
    
    std::vector<std::vector<double>> weights = layer->getWeights();
    std::vector<std::vector<double>> dweights = layer->getDweights();
    std::vector<std::vector<double>> biases = layer->getBiases();
    std::vector<std::vector<double>> dbiases = layer->getDbiases();

    std::vector<std::vector<double>> weight_momentums_corrected(weight_momentums.size(), std::vector<double>(weight_momentums[0].size(), 0.0));
    std::vector<std::vector<double>> bias_momentums_corrected(bias_momentums.size(), std::vector<double>(bias_momentums[0].size(), 0.0));
    std::vector<std::vector<double>> weight_cache_corrected(weight_cache.size(), std::vector<double>(weight_cache[0].size(), 0.0));
    std::vector<std::vector<double>> bias_cache_corrected(bias_cache.size(), std::vector<double>(bias_cache[0].size(), 0.0));

    for (int i = 0; i < weight_momentums.size(); ++i) {
        for (int j = 0; j < weight_momentums[0].size(); ++j) { 
            weight_momentums[i][j] = beta_1 * weight_momentums[i][j] + ( 1 - beta_1) * dweights[i][j];
        }
    }

    for (int i = 0; i < bias_momentums.size(); ++i) {
        for (int j = 0; j < bias_momentums[0].size(); ++j) { 
            bias_momentums[i][j] = beta_1 * bias_momentums[i][j] + ( 1 - beta_1) * dbiases[i][j];
        }
    }

    for (int i = 0; i < weight_momentums_corrected.size(); ++i) {
        for (int j = 0; j < weight_momentums_corrected[0].size(); ++j) { 
            weight_momentums_corrected[i][j] =  weight_momentums[i][j] / std::pow( 1 - beta_1 , (iterations+1));
        }
    }

    for (int i = 0; i < bias_momentums_corrected.size(); ++i) {
        for (int j = 0; j < bias_momentums_corrected[0].size(); ++j) { 
            bias_momentums_corrected[i][j] = bias_momentums[i][j] / std::pow( 1 - beta_1 , (iterations+1));
        }
    }

     for (int i = 0; i < weight_cache.size(); ++i) {
        for (int j = 0; j < weight_cache[0].size(); ++j) { 
            weight_cache[i][j] = beta_2 * weight_cache[i][j] + ( 1 - beta_2) * std::pow(dweights[i][j] , 2);
        }
    }

    for (int i = 0; i < bias_cache.size(); ++i) {
        for (int j = 0; j < bias_cache[0].size(); ++j) { 
            bias_cache[i][j] = beta_2 * bias_cache[i][j] + ( 1 - beta_2) * std::pow(dbiases[i][j] , 2);
        }
    }

    for (int i = 0; i < weight_cache_corrected.size(); ++i) {
        for (int j = 0; j < weight_cache_corrected[0].size(); ++j) { 
            weight_cache_corrected[i][j] =  weight_cache[i][j] / std::pow( 1 - beta_2 , (iterations+1));
        }
    }

    for (int i = 0; i < bias_cache_corrected.size(); ++i) {
        for (int j = 0; j < bias_cache_corrected[0].size(); ++j) { 
            bias_cache_corrected[i][j] = bias_cache[i][j] / std::pow( 1 - beta_2 , (iterations+1));
        }
    }


     for (int i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < weights[0].size(); ++j) { 
            weights[i][j] += (- current_learning_rate * weight_momentums_corrected[i][j]) / (std::sqrt(weight_cache_corrected[i][j]) + epsilon);
        }
    }
    
    
    for (int i = 0; i < biases.size(); ++i) {
        for (int j = 0; j < biases[0].size(); ++j) { 
            biases[i][j] += (- current_learning_rate * bias_momentums_corrected[i][j]) / (std::sqrt(bias_cache_corrected[i][j]) + epsilon);
        }
    }

    layer->setWeights(weights);
    layer->setBiases(biases);
    layer->setWeightMomentums(weight_momentums);
    layer->setBiasMomentums(bias_momentums);
    layer->setWeightCache(weight_cache);
    layer->setBiasCache(bias_cache);
 

}

void Adam::post_update_params(){
   iterations++;
}