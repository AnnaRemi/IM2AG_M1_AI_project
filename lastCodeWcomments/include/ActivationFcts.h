#ifndef ACTIVATIONFCTS_H
#define ACTIVATIONFCTS_H

#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>

#include "Matrix.h"

/**
 * @class ActivationFcts
 * @brief Provides implementations of common activation functions and their derivatives used in neural networks.
 */
class ActivationFcts
{
public:
    /**
     * @brief Applies the ReLU activation function to a vector.
     * @param inputs A vector containing the input values.
     * @return A vector where each element is the result of applying ReLU.
     */
    std::vector<double> ReLU(const std::vector<double> &inputs);

    /**
     * @brief Applies the ReLU activation function to a matrix.
     * @param input A shared pointer to a Matrix object containing the input values.
     * @return A shared pointer to a Matrix object with ReLU applied element-wise.
     */
    std::shared_ptr<Matrix> ReLU(std::shared_ptr<Matrix> input);

    /**
     * @brief Computes the derivative of the ReLU activation function.
     * @param dvalues A shared pointer to a Matrix containing the gradient of the loss with respect to the outputs.
     * @param inputs A shared pointer to a Matrix containing the input values.
     * @return A shared pointer to a Matrix containing the gradients after applying the derivative of ReLU.
     */
    std::shared_ptr<Matrix> reluDerivative(std::shared_ptr<Matrix> dvalues, std::shared_ptr<Matrix> inputs);

    /**
     * @brief Applies the Softmax activation function to a matrix.
     * @param input A shared pointer to a Matrix containing the input values.
     * @return A shared pointer to a Matrix with Softmax applied row-wise.
     * @details The Softmax function converts raw scores into probabilities by exponentiating and normalizing each row.
     */
    std::shared_ptr<Matrix> Softmax(std::shared_ptr<Matrix> input);
};

#endif // ACTIVATIONFCTS_H
