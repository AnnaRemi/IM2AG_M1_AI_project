#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <vector>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <memory>

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
     * @param layer Shared pointer to the layer whose parameters are to be updated.
     * @param num_l Integer representing the layer number (useful for debugging).
     */
    virtual void update_params(std::shared_ptr<Layer> layer, int num_l) = 0;
};

/**
 * @class Adam
 * @brief Implementation of the Adam optimization algorithm.
 */
class Adam : public Optimizers
{
private:
    double learning_rate;         ///< The learning rate used for updates.
    double current_learning_rate; ///< Current learning rate after applying decay.
    double decay;                 ///< Learning rate decay factor.
    int iterations;               ///< Number of iterations for decay and momentum correction.
    double epsilon;               ///< A small value to avoid division by zero.
    double beta_1;                ///< Exponential decay rate for the first moment estimates.
    double beta_2;                ///< Exponential decay rate for the second moment estimates.

public:
    /**
     * @brief Constructs an Adam optimizer with specified parameters.
     * @param lr Learning rate.
     * @param dec Learning rate decay.
     * @param ep Epsilon value to avoid division by zero.
     * @param b1 Beta_1, the exponential decay rate for the first moment estimates.
     * @param b2 Beta_2, the exponential decay rate for the second moment estimates.
     */
    Adam(double lr, double dec, double ep, double b1, double b2);

    /**
     * @brief Prepares the optimizer for parameter updates (applies decay if specified).
     */
    void pre_update_params();

    /**
     * @brief Updates the parameters (weights and biases) of a given layer using Adam optimization.
     * @param layer Shared pointer to the layer whose parameters are to be updated.
     * @param num_l Integer representing the layer number.
     */
    void update_params(std::shared_ptr<Layer> layer, int num_l);

    /**
     * @brief Finalizes updates (increments iteration counter).
     */
    void post_update_params();

    /**
     * @brief Gets the current learning rate after decay.
     * @return Current learning rate.
     */
    double getCurrent_learning_rate();
};

/**
 * @class GradientDescent
 * @brief Implementation of the Gradient Descent optimization algorithm.
 */
class GradientDescent : public Optimizers
{
private:
    double learning_rate; ///< The learning rate used for updates.

public:
    /**
     * @brief Constructs a GradientDescent optimizer with a specified learning rate.
     * @param lr Learning rate for the optimizer.
     */
    GradientDescent(double lr);

    /**
     * @brief Updates the parameters (weights and biases) of a given layer using gradient descent.
     * @param layer Shared pointer to the layer whose parameters are to be updated.
     * @param num_l Integer representing the layer number.
     */
    void update_params(std::shared_ptr<Layer> layer, int num_l);
};

/**
 * @class GradientDescentWithDecay
 * @brief Implementation of Gradient Descent with learning rate decay.
 */
class GradientDescentWithDecay : public Optimizers
{
private:
    double learning_rate;         ///< Initial learning rate.
    double current_learning_rate; ///< Current learning rate after applying decay.
    double decay;                 ///< Learning rate decay factor.
    int iterations;               ///< Number of iterations for decay application.

public:
    /**
     * @brief Constructs a GradientDescentWithDecay optimizer with specified parameters.
     * @param learning_rate Initial learning rate.
     * @param decay Learning rate decay factor.
     */
    GradientDescentWithDecay(double learning_rate, double decay);

    /**
     * @brief Prepares the optimizer for parameter updates (applies decay if specified).
     */
    void pre_update_params();

    /**
     * @brief Updates the parameters (weights and biases) of a given layer using gradient descent with decay.
     * @param layer Shared pointer to the layer whose parameters are to be updated.
     * @param num_l Integer representing the layer number.
     */
    void update_params(std::shared_ptr<Layer> layer, int num_l);

    /**
     * @brief Finalizes updates (increments iteration counter).
     */
    void post_update_params();
};

/**
 * @class RandomUpdate
 * @brief Implementation of a random update for weights and biases.
 */
class RandomUpdate : public Optimizers
{
public:
    /**
     * @brief Updates the parameters (weights and biases) of a given layer at random.
     * @param layer Shared pointer to the layer whose parameters are to be updated.
     * @param num_l Integer representing the layer number.
     */
    void update_params(std::shared_ptr<Layer> layer, int num_l);
};

#endif // OPTIMIZERS_H
 