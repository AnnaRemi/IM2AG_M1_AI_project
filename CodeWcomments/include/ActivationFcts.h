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
 * @brief Class implementing various activation functions used in the neural network.
 */
class ActivationFcts
{
public:
    /**
     * @brief Applies the ReLU activation function to a vector.
     * @param inputs A vector of input values.
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
     * @brief Computes the derivative of the ReLU function.
     * @param dvalues A shared pointer to a Matrix containing the gradients.
     * @param inputs A shared pointer to a Matrix containing the input values.
     * @return A shared pointer to a Matrix containing the gradients after applying ReLU derivative.
     */
    std::shared_ptr<Matrix> reluDerivative(std::shared_ptr<Matrix> dvalues, std::shared_ptr<Matrix> inputs);

    /**
     * @brief Applies the Softmax activation function to a matrix.
     * @param input A shared pointer to a Matrix containing the input values.
     * @return A shared pointer to a Matrix where Softmax is applied row-wise.
     */
    std::shared_ptr<Matrix> Softmax(std::shared_ptr<Matrix> input);
};

#endif // ACTIVATIONFCTS_H