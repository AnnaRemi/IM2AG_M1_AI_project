#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <memory>
#include "Matrix.h"

class Layer{
    private:
        int numInputs;
        int numNeurons;
        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> weights;
        std::vector<std::vector<double>> biases;
        //std::shared_ptr<Matrix> output;
        std::vector<std::vector<double>> output;
        std::vector<std::vector<double>> dinputs;
        std::vector<std::vector<double>> dweights;
        std::vector<std::vector<double>> dbiases;

    public:
        Layer( int ni,int nn, std::vector<std::vector<double>> w);
        Layer( int ni, int nn);
        Layer(std::vector<std::vector<double>> w);
        
        //Setters
        void setWeights(const std::vector<std::vector<double>> &);
        void setBiases(const std::vector<std::vector<double>> &);

        //Getters
        std::vector<std::vector<double>> getWeights() const;
        Matrix getWeights_mat() const;
        std::vector<std::vector<double>> getBiases() const;
        std::vector<std::vector<double>> getDweights() const;
        std::vector<std::vector<double>> getDbiases() const;
        //std::shared_ptr<Matrix> getOutput() const;
        std::vector<std::vector<double>> getOutput() const;
        std::vector<std::vector<double>> getDinputs() const;
        void printMatrix(std::vector<std::vector<double>> matrix);

        void multiply(std::shared_ptr<Matrix> mat_weights, std::shared_ptr<Matrix> mat_inputs);
        void add(std::shared_ptr<Matrix> mat_outputs);
        void forward(std::vector<std::vector<double>> inputs); 
        void backward(std::vector<std::vector<double>> dvalues);    

};

#endif // LAYER_H