#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include "Neuron.h"
#include "Matrix.h"

class Optimizer;

class Layer{
    friend class Optimizer;

private:
    int numInputs;
    int numNeurons;
    Matrix inputs;
    Matrix weights;
    Matrix biases;
    Matrix output;
    Matrix dinputs;
    Matrix dweights;
    Matrix dbiases;

public:
    Layer(int ni, int nn);
    Layer(int ni, int nn, std::vector<std::vector<double>> m, std::vector<std::vector<double>> b);
    Layer(int ni, int nn, std::vector<std::vector<double>> b);

    Matrix getOutput() const;
    Matrix get_dinputs() const;
    Matrix get_dweights() const;
    Matrix get_dbiases() const;
    void multiply();
    void add();
    void forward(std::vector<std::vector<double>> inputs);
    void backward(std::vector<std::vector<double>> dvalues);    

};


class Optimizer{
    private:
        double learning_rate;
    public:
        Optimizer(double l=1);
        void update_parameters(Layer);
};

#endif // LAYER_H
