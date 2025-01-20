#include "../include/LossFcts.h"
#include "../include/Matrix.h"
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <memory>

double clip(const double &value, const double &min, const double &max) {
   return std::max(min,std::min(value,max));
}

std::vector<double> clip(const std::vector<double> &values, const double &min, const double &max) {
   std::vector<double> result(values.size());

   std::transform(values.begin(), values.end(), result.begin(), [&min,&max](const double& val){return clip(val,min,max);});

   return result;
}

std::shared_ptr<Matrix> clip(const std::vector<std::vector<double>> &matrix, const double &min, const double &max) {
    std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));

    for (int i = 0; i < matrix.size(); ++i) {
            auto part_result = clip(matrix[i], min, max);
            for (int j = 0; j < matrix[0].size(); ++j) {
                result[i][j] = part_result[j];
            }
    }

    std::shared_ptr<Matrix> result_mat = std::make_shared<Matrix>(result.size(), result[0].size(),result);
    return result_mat;
}

std::vector<double> negative_log_likelihood(const std::vector<double>& vec){
    std::vector<double> result;
    for (int i = 0; i < vec.size(); ++i) {
        result.push_back(-std::log(vec[i]));
    }
    return result;
}

double mean_vector(const std::vector<double>& vec){
    double sum = 0;
    for (int i = 0; i < vec.size(); ++i) {
        sum += vec[i];
    }
    return sum / vec.size();
}


//Cross entropy loss function forward method
double LossFcts::crossEntropyLoss_forward(std::shared_ptr<Matrix> predictions, std::shared_ptr<Matrix> targets) {
    std::vector<double> correct_confidences;
    std::shared_ptr<Matrix> clipped_prediction = clip(predictions->getValues(), 1e-7, 1 - 1e-7);
    
    //std::cout << (targets->getNumRows() == predictions->getNumRows()) << std::endl;
    int samples = targets->getNumRows();
    //<=> if len(y_true.shape)==1
    if (targets->getNumCols() == 1)
    {
        // Categorical labels
        for (int i = 0; i < samples; ++i)
        {
            int correctClass = static_cast<int>(targets->getValue(i, 0)); // Get the true class index
            correct_confidences.push_back(clipped_prediction->getValue(i, correctClass));
        }
    }
    else
    {
        // One-hot encoded labels
        for (int i = 0; i < samples; ++i)
        {
            double confidence = 0.0;
            for (int j = 0; j < clipped_prediction->getNumCols(); ++j)
            {
                confidence += clipped_prediction->getValue(i, j) * targets->getValue(i, j);
            }
            correct_confidences.push_back(confidence);
        }
    }

    // Compute negative log likelihoods
    std::vector<double> negativeLogLikelihoods;
    for (double confidence : correct_confidences)
    {
        negativeLogLikelihoods.push_back(-std::log(confidence));
    }

    // Create a Matrix to store the loss values and return it
    std::shared_ptr<Matrix> lossMatrix = std::make_shared<Matrix>(samples, 1);
    for (int i = 0; i < samples; ++i)
    {
        lossMatrix->setValue(i, 0, negativeLogLikelihoods[i]);
    }
    // Calculate the mean loss
    double totalLoss = 0.0;
    for (double loss : negativeLogLikelihoods)
    {
        totalLoss += loss;
    }
    double meanLoss = totalLoss / static_cast<double>(samples);
    return meanLoss;
}

//Cross entropy loss function backward method
std::shared_ptr<Matrix> LossFcts::crossEntropyLoss_backward_softmax(std::shared_ptr<Matrix> dvalues, std::shared_ptr<Matrix> targets) {
    std::shared_ptr<Matrix> dinputs = dvalues->copy();
    int samples = dvalues->getNumRows(); // Number of samples in the batch
    int labels = dvalues->getNumCols();  // Number of labels per sample
    if(targets->getNumCols() == 1){
        for (int i = 0; i < samples; ++i)
        {
            int correctClass = static_cast<int>(targets->getValue(i, 0)); // True class index
            dinputs->setValue(i, correctClass, dinputs->getValue(i, correctClass) - 1);
        }
    }
    else
    {
        // If targets are one-hot encoded, subtract directly
        for (int i = 0; i < samples; ++i)
        {
            for (int j = 0; j < labels; ++j)
            {
                dinputs->setValue(i, j, dinputs->getValue(i, j) - targets->getValue(i, j));
            }
        }
    }

    return dinputs->divide(samples);
}

