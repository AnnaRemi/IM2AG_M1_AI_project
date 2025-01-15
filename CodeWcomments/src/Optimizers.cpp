#include "../include/Optimizers.h"

GradientDescent::GradientDescent(double lr){
    learning_rate = lr;
}

void GradientDescent::update_params(Layer &layer){
    std::vector<std::vector<double>> weights = layer.getWeights();
    std::vector<std::vector<double>> dweights = layer.getDweights();
    std::vector<std::vector<double>> biases = layer.getBiases();
    std::vector<std::vector<double>> dbiases = layer.getDbiases();

    
    for (int i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < weights[0].size(); ++j) { 
            weights[i][j] = - learning_rate * dweights[i][j];
        }
    }
    
    
    for (int i = 0; i < biases.size(); ++i) {
        for (int j = 0; j < biases[0].size(); ++j) { 
            biases[i][j] = - learning_rate * dbiases[i][j];
        }
    }
    
    layer.setWeights(weights);
    layer.setBiases(biases);
}