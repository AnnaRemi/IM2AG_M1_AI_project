/**
 * @file Layer_version3.hpp
 * @brief Version with eigen library
 * @version 3
 */

#ifndef LAYER_VERSION3_HPP
#define LAYER_VERSION3_HPP

#include <vector>
#include <iostream>
#include <ctime>
#include "lib/eigen-3.4.0/Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
/**
 * @class Layer
 * @brief Class Layer using the eigen library.
 *
 * A `Layer` has two components :
 * - `neurons`: A matrix of size `n x w`, where `n` is the number of neurons in the layer, 
 * and `w` is the number of weights (inputs) each neuron has.
 * - `biases`: A vector of size `n` representing the biases associated with each neuron in the layer.
 * 
 */
class Layer{
    private :
        MatrixXd neurons; /**< matrix where each column represents the weights of a neuron*/
        VectorXd biases;/**< vector containing the bias of each neuron */

    public :
        /**
         * @brief Constructs a new Layer object by initializing neurons a matrix of size n_inout*n_neurons with random coefficients 
         * and biases a vector of size n_neurons with all the coefficients equal to zero
         * @param n_input The number of input each neuron can take (= to the number of weights of each neuron)
         * @param n_neurons Number of neurons of the Layer
         */
        Layer(int n_input, int n_neurons);

        /**
         * @brief Constructs a Layer object with specified neuron weights and biases.
         *
         * This constructor initializes the `neurons` matrix with the provided matrix `n` and the `biases` vector
         * with the provided vector `b`. If the size of the biases vector does not match the number of columns
         * in the `neurons` matrix, an exception is thrown.
         *
         * @param n A matrix representing the weights of the neurons. Size: `n_input x n_neurons`.
         * @param b A vector representing the biases for each neuron. Size: `n_neurons`.
         * @throws std::invalid_argument if the number of elements in `b` does not match the number of neurons.
         */
        Layer(MatrixXd n, VectorXd b);

        /**
         * @brief Performs the forward propagation of the layer with a given input
         * 
         * The input has to be a MatrixXd of size b*w where w is the number of weights of the neurons 
         * (number of columns of the matrix neurons of this Layer)
         * @param inputs A matrix of size `b x w`, where `b` is the number of input samples and `w` is the number of weights of each neuron.
         * @return MatrixXd A matrix of size `n x b`, where `n` is the number of neurons, and `b` is the number of input samples.
         * @throws std::invalid_argument if the number of columns of the input matrix does not match the number 
         * of rows of the `neurons` matrix.
         * 
         */
        MatrixXd forward(MatrixXd inputs);

        friend std::ostream &operator<<(std::ostream &o, const Layer &L);
};

/**
 * @brief Performs the activation_ReLU to the given matrix input and returns the new matrix
 *
 * @param input The given matrix
 * @return MatrixXd a matrix of same dimension as the input matrix,
 *  with each coefficient being max(0, input(i,j)).
 */
MatrixXd activation_ReLU(MatrixXd input);

/**
 * @brief Performs the activation softmax to the given mmatrix input and returns the new matrix
 * 
 * @param input The given matrix
 * @return MatrixXd a matrix of same dimension as the input matrix,
 * with each coefficients being exp(input(i,j))/(exp(input(i,0))+...+exp(input(i,k)))
 */
MatrixXd activation_softmax(MatrixXd input);

/**
 * @brief 
 * 
 * @param y_pred 
 * @param y_true 
 * @return MatrixXd 
 */
MatrixXd loss(MatrixXd y_pred, MatrixXd y_true);

#endif