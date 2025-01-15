#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <memory>
#include "Matrix.h"

/**
 * @class Layer
 * @brief Represents a neural network layer with forward and backward propagation
 */
class Layer
{
private:
    int numInputs;                             ///< Number of inputs of each neurons of the layer
    int numNeurons;                            ///< Number of neurons in the layer
    std::vector<std::vector<double>> inputs;   ///< Matrix (m x n) storing input values, where m is the batch size(number of batch of inputs) and n is numInputs
    std::vector<std::vector<double>> weights;  ///< Matrix (numInputs x numNeurons) storing weight values
    std::vector<std::vector<double>> biases;   ///< Matrix (1 x numNeurons) storing bias
    std::vector<std::vector<double>> output;   ///< Matrix (m x numNeurons) storing output values after forward pass
    std::vector<std::vector<double>> dinputs;  ///< Matrix (m x numInputs) storing gradients with respect to inputs
    std::vector<std::vector<double>> dweights; ///< Matrix (numInputs x numNeurons) storing gradients with respect to weights.
    std::vector<std::vector<double>> dbiases;  ///< Matrix (1 x numNeurons) storing gradients with respect to biases.

public:
    /**
     * @brief Constructs a layer with given dimensions and initial weights.
     * @param ni Number of inputs.
     * @param nn Number of neurons.
     * @param m Weight matrix (numInputs x numNeurons).
     */
    Layer(int ni, int nn, std::vector<std::vector<double>> m);

    /**
     * @brief Constructs a layer with given dimensions and zero-initialized weights.
     * @param ni Number of inputs.
     * @param nn Number of neurons.
     */
    Layer(int ni, int nn);

    /**
     * @brief Sets the weights of the layer.
     * @param w Matrix (numInputs x numNeurons) containing the new weights.
     */
    void setWeights(std::vector<std::vector<double>> w);

    /**
     * @brief Sets the biases of the layer.
     * @param b Matrix (1 x numNeurons) containing the new biases.
     */
    void setBiases(std::vector<std::vector<double>> b);

    /**
     * @brief Gets the current weights of the layer.
     * @return Matrix (numInputs x numNeurons) representing the current weights.
     */
    std::vector<std::vector<double>> getWeights() const;

    /**
     * @brief Gets the current biases of the layer.
     * @return Matrix (1 x numNeurons) representing the current biases.
     */
    std::vector<std::vector<double>> getBiases() const;

    /**
     * @brief Gets the gradients with respect to weights.
     * @return Matrix (numInputs x numNeurons) representing the gradients.
     */
    std::vector<std::vector<double>> getDweights() const;

    /**
     * @brief Gets the gradients with respect to biases.
     * @return Matrix (1 x numNeurons) representing the gradients.
     */
    std::vector<std::vector<double>> getDbiases() const;

    /**
     * @brief Gets the output matrix of the layer.
     * @return Matrix (m x numNeurons) representing the output.
     */
    std::vector<std::vector<double>> getOutput() const;

    /**
     * @brief Gets the gradients with respect to inputs.
     * @return Matrix (m x numInputs) representing the gradients.
     */
    std::vector<std::vector<double>> getDinputs() const;

    /**
     * @brief Adds the bias matrix to the outputs.
     * @param mat_outputs Shared pointer to the output matrix (m x numNeurons).
     */
    void add(std::shared_ptr<Matrix> mat_outputs);

    /**
     * @brief Performs forward propagation.
     * @param inputs Matrix (m x numInputs) representing the input values.
     */
    void forward(std::vector<std::vector<double>> inputs);

    /**
     * @brief Performs backward propagation.
     * @param dvalues Matrix (m x numNeurons) representing the gradients with respect to outputs.
     */
    void backward(std::vector<std::vector<double>> dvalues);
};

#endif // LAYER_H