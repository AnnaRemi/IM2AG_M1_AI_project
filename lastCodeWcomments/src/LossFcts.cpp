/**
 * @file LossFcts.cpp
 * @brief Contains the definitions of the functions declared in LossFcts.h
 */

#include "../include/LossFcts.h"
#include "../include/Matrix.h"
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <memory>

/**
 * @brief Clips a given value to a specified range.
 *
 * @param value The value to be clipped.
 * @param min The minimum allowable value.
 * @param max The maximum allowable value.
 * @return The clipped value within the range [min, max].
 */
double clip(const double &value, const double &min, const double &max)
{
    return std::max(min, std::min(value, max));
}

/**
 * @brief Clips each element of a vector to a specified range.
 *
 * @param values The vector of values to be clipped.
 * @param min The minimum allowable value for each element.
 * @param max The maximum allowable value for each element.
 * @return A vector with each element clipped within the range [min, max].
 */
std::vector<double> clip(const std::vector<double> &values, const double &min, const double &max)
{
    std::vector<double> result(values.size());

    std::transform(values.begin(), values.end(), result.begin(), [&min, &max](const double &val)
                   { return clip(val, min, max); });

    return result;
}

/**
 * @brief Clips each element of a matrix (2D vector) to a specified range.
 *
 * @param matrix The 2D vector (matrix) to be clipped.
 * @param min The minimum allowable value for each element in the matrix.
 * @param max The maximum allowable value for each element in the matrix.
 * @return A shared pointer to a Matrix object containing the clipped values.
 *
 */
std::shared_ptr<Matrix> clip(const std::vector<std::vector<double>> &matrix, const double &min, const double &max)
{
    std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));

    for (int i = 0; i < matrix.size(); ++i)
    {
        auto part_result = clip(matrix[i], min, max);
        for (int j = 0; j < matrix[0].size(); ++j)
        {
            result[i][j] = part_result[j];
        }
    }

    std::shared_ptr<Matrix> result_mat = std::make_shared<Matrix>(result.size(), result[0].size(), result);
    return result_mat;
}

/**
 * @brief Computes the negative log-likelihood for each element in a vector.
 *
 * @param vec The input vector of probabilities.
 * @return A vector containing the negative log-likelihood for each element in the input vector.
 *
 * @note The input vector elements must be positive, as log is undefined for non-positive values.
 */
std::vector<double> negative_log_likelihood(const std::vector<double> &vec)
{
    std::vector<double> result;
    for (int i = 0; i < vec.size(); ++i)
    {
        result.push_back(-std::log(vec[i]));
    }
    return result;
}

/**
 * @brief Calculates the mean of a vector of values.
 *
 * @param vec The input vector of values.
 * @return The mean of the values in the input vector.
 */
double mean_vector(const std::vector<double> &vec)
{
    double sum = 0;
    for (int i = 0; i < vec.size(); ++i)
    {
        sum += vec[i];
    }
    return sum / vec.size();
}


//Cross entropy loss function forward method

double LossFcts::crossEntropyLoss_forward(std::shared_ptr<Matrix> predictions, std::shared_ptr<Matrix> targets) {
    std::vector<double> correct_confidences;
    std::shared_ptr<Matrix> clipped_prediction = clip(predictions->getValues(), 1e-7, 1 - 1e-7);
    
    if(targets->getValues().size() == 1){
        for (int j = 0; j < targets->getNumCols(); j++) { 
            correct_confidences.push_back(predictions->getValue(j, targets->getValue(0,j)));
        } 
    }
    else{
        std::shared_ptr<Matrix> mat_mult = (*clipped_prediction) * (*targets);
        correct_confidences = mat_mult->sumMat();
    }

    std::vector<double> nll = negative_log_likelihood(correct_confidences);
    
   
    
    return mean_vector(nll);
}



//Cross entropy loss function backward method

std::shared_ptr<Matrix> LossFcts::crossEntropyLoss_backward_softmax(std::shared_ptr<Matrix> dvalues, std::shared_ptr<Matrix> targets) {
    std::shared_ptr<Matrix> dinputs = dvalues->copy();

    /*for (int j = 295; j < dvalues->getNumRows(); j++) { 
        int target_idx = targets->getValue(j,0);
        std::cout << "target_idx" << target_idx << std::endl;
    } */

    for (int j = 0; j < dvalues->getNumRows(); j++) { 
        int target_idx = targets->getValue(j,0);
        dinputs->setValue(j, target_idx, dinputs->getValue(j, target_idx) - 1);
    } 
    
    return dinputs->divide(dvalues->getNumRows());
}

