#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <vector>
#include <math.h>
#include <iostream>
#include <cstdlib>

#include "Matrix.h"
#include "Layer.h"

/**
 * @class Optimizers
 * @brief Abstract base class for optimizers used in neural network training.
 */
class Optimizers
{
public:
    /**
     * @brief Pure virtual method for updating parameters of a layer.
     * @param layer Reference to the layer whose parameters are to be updated.
     */
    virtual void update_params(Layer &) = 0;
};

/**
 * @class GradientDescent
 * @brief Implementation of the Gradient Descent optimization algorithm.
 */
class GradientDescent : public Optimizers
{
private:
    double learning_rate; ///< The learning rate used for the gradient descent update.

public:
    /**
     * @brief Constructs a GradientDescent optimizer with a specified learning rate.
     * @param lr The learning rate for the optimizer.
     */
    GradientDescent(double lr);

    /**
     * @brief Updates the parameters (weights and biases) of a given layer using gradient descent.
     * @param layer Reference to the layer whose parameters are to be updated.
     * @details The parameters are updated using the formula:
     *          weights -= learning_rate * dweights;
     *          biases -= learning_rate * dbiases;
     */
    void update_params(Layer &layer);
};

#endif // OPTIMIZERS_H