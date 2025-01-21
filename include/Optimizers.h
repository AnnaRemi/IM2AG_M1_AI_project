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
class Optimizers {
    public:
        /**
         * @brief Pure virtual method for updating parameters of a layer.
         * @param layer Reference to the layer whose parameters are to be updated.
         */
        virtual void update_params(std::shared_ptr<Layer> layer) = 0;
};

/**
 * @class Adam
 * @brief Implementation of the Gradient Descent optimization algorithm.
 */
class Adam: public Optimizers{
    private:
        double learning_rate; ///< The learning rate used for the gradient descent update.
        double current_learning_rate;
        double decay;
        int iterations;
        double epsilon;
        double beta_1;
        double beta_2;
    public:
        /**
         * @brief Constructs a Adam optimizer with a specified learning rate.
         * @param lr The learning rate for the optimizer.
         */
        Adam(double lr, double dec, double ep, double b1, double b2);

        /**
         * @brief Updates the parameters (weights and biases) of a given layer using gradient descent.
         * @param layer Shared pointer to the layer whose parameters are to be updated.
         * @details The parameters are updated using the formula:
         *          weights += -learning_rate * dweights;
         *          biases += -learning_rate * dbiases;
         */
        void pre_update_params();
        void update_params(std::shared_ptr<Layer> layer);
        void post_update_params();
        double getCurrent_learning_rate();

};

/**
 * @class GradientDescent
 * @brief Implementation of the Gradient Descent optimization algorithm.
 */
class GradientDescent: public Optimizers{
    private:
        double learning_rate; ///< The learning rate used for the gradient descent update.
    public:
        /**
         * @brief Constructs a GradientDescent optimizer with a specified learning rate.
         * @param lr The learning rate for the optimizer.
         */
        GradientDescent(double);

        /**
         * @brief Updates the parameters (weights and biases) of a given layer using gradient descent.
         * @param layer Shared pointer to the layer whose parameters are to be updated.
         * @details The parameters are updated using the formula:
         *          weights += -learning_rate * dweights;
         *          biases += -learning_rate * dbiases;
         */
        void update_params(std::shared_ptr<Layer> layer);
};


/**
 * @class RandomUpdate
 * @brief Implementation of the random update of weights and biases.
 */
class RandomUpdate: public Optimizers{
    public:

        /**
         * @brief Updates the parameters (weights and biases) of a given layer at random.
         * @param layer Shared pointer to the layer whose parameters are to be updated.
         */
        void update_params(std::shared_ptr<Layer> layer);
};


#endif // OPTIMIZERS_H