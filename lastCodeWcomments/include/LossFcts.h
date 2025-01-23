#ifndef LOSSFCTS_H
#define LOSSFCTS_H

#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>
#include "Matrix.h"

/**
 * @class LossFcts
 * @brief Implements loss functions for neural networks.
 */
class LossFcts
{
public:
    /**
     * @brief Calculates the forward pass of the cross-entropy loss function.
     * @param predictions A shared pointer to a Matrix containing the predicted values (size: batch_size x num_classes).
     * @param targets A shared pointer to a Matrix containing the true values (size: batch_size x 1 or batch_size x num_classes for one-hot encoding).
     * @return The computed cross-entropy loss as a double.
     */
    double crossEntropyLoss_forward(std::shared_ptr<Matrix> predictions, std::shared_ptr<Matrix> targets);

    /**
     * @brief Calculates the backward pass of the cross-entropy loss function combined with the softmax activation.
     * @param dvalues A shared pointer to a Matrix containing the gradients of the loss with respect to outputs (size: batch_size x num_classes).
     * @param targets A shared pointer to a Matrix containing the true labels (size: batch_size x 1 or batch_size x num_classes for one-hot encoding).
     * @return A shared pointer to a Matrix containing the gradients of the loss with respect to inputs (size: batch_size x num_classes).
     */
    std::shared_ptr<Matrix> crossEntropyLoss_backward_softmax(std::shared_ptr<Matrix> dvalues, std::shared_ptr<Matrix> targets);
};

#endif // LOSSFCTS_H
