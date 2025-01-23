#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <memory>
#include "Matrix.h"

/**
 * @class Layer
 * @brief Represents a neural network layer with forward and backward propagation.
 */
class Layer
{
private:
    int numInputs;                             ///< Number of inputs to each neuron of the layer.
    int numNeurons;                            ///< Number of neurons in the layer.
    std::vector<std::vector<double>> inputs;   ///< Matrix (m x n) storing input values, where m is the batch size (number of input samples) and n is numInputs.
    std::vector<std::vector<double>> weights;  ///< Matrix (numInputs x numNeurons) storing weight values.
    std::vector<std::vector<double>> biases;   ///< Matrix (1 x numNeurons) storing biases.
    std::vector<std::vector<double>> output;   ///< Matrix (m x numNeurons) storing output values after forward pass.
    std::vector<std::vector<double>> dinputs;  ///< Matrix (m x numInputs) storing gradients with respect to inputs.
    std::vector<std::vector<double>> dweights; ///< Matrix (numInputs x numNeurons) storing gradients with respect to weights.
    std::vector<std::vector<double>> dbiases;  ///< Matrix (1 x numNeurons) storing gradients with respect to biases.

    // Attributes for Adam optimizer
    std::vector<std::vector<double>> weight_momentums; ///< Weight momentums for Adam optimizer.
    std::vector<std::vector<double>> weight_cache;     ///< Weight cache for Adam optimizer.
    std::vector<std::vector<double>> bias_momentums;   ///< Bias momentums for Adam optimizer.
    std::vector<std::vector<double>> bias_cache;       ///< Bias cache for Adam optimizer.

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

    // Setters

    /**
     * @brief Sets the weights of the layer.
     * @param w Matrix (numInputs x numNeurons) containing the new weights.
     */
    void setWeights(const std::vector<std::vector<double>> &w);

    /**
     * @brief Sets the biases of the layer.
     * @param b Matrix (1 x numNeurons) containing the new biases.
     */
    void setBiases(const std::vector<std::vector<double>> &b);

    /**
     * @brief Sets the weight momentums of the layer (used for optimizers).
     * @param wm Matrix (numInputs x numNeurons) containing the weight momentums.
     */
    void setWeightMomentums(const std::vector<std::vector<double>> &wm);

    /**
     * @brief Sets the bias momentums of the layer (used for optimizers).
     * @param bm Matrix (1 x numNeurons) containing the bias momentums.
     */
    void setBiasMomentums(const std::vector<std::vector<double>> &bm);

    /**
     * @brief Sets the weight cache of the layer (used for optimizers).
     * @param wc Matrix (numInputs x numNeurons) containing the weight cache.
     */
    void setWeightCache(const std::vector<std::vector<double>> &wc);

    /**
     * @brief Sets the bias cache of the layer (used for optimizers).
     * @param bc Matrix (1 x numNeurons) containing the bias cache.
     */
    void setBiasCache(const std::vector<std::vector<double>> &bc);

    // Getters

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
     * @brief Gets the weight momentums of the layer (used for optimizers).
     * @return Matrix (numInputs x numNeurons) representing the weight momentums.
     */
    std::vector<std::vector<double>> getWeightMomentums() const;

    /**
     * @brief Gets the bias momentums of the layer (used for optimizers).
     * @return Matrix (1 x numNeurons) representing the bias momentums.
     */
    std::vector<std::vector<double>> getBiasMomentums() const;

    /**
     * @brief Gets the weight cache of the layer (used for optimizers).
     * @return Matrix (numInputs x numNeurons) representing the weight cache.
     */
    std::vector<std::vector<double>> getWeightCache() const;

    /**
     * @brief Gets the bias cache of the layer (used for optimizers).
     * @return Matrix (1 x numNeurons) representing the bias cache.
     */
    std::vector<std::vector<double>> getBiasCache() const;

    // Utility Methods

    /**
     * @brief Prints the provided matrix.
     * @param matrix Vector of vectors to be printed.
     */
    void printMatrix(std::vector<std::vector<double>> matrix);

    /**
     * @brief Adds the bias matrix to the outputs.
     * @param mat_outputs Shared pointer to the output matrix (m x numNeurons).
     */
    void add(std::shared_ptr<Matrix> mat_outputs);

    /**
     * @brief Performs forward propagation on the current Layer with the given inputs
     * @param inputs Matrix (m x numInputs) representing the input values.
     */
    void forward(std::vector<std::vector<double>> inputs);

    /**
     * @brief Performs backward propagation on the current Layer
     * @param dvalues Matrix (m x numNeurons) representing the gradients with respect to outputs.
     */
    void backward(std::vector<std::vector<double>> dvalues);
};

#endif // LAYER_H
