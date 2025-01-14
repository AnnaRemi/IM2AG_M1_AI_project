#include "../include/LossFcts.h"
#include "../include/Matrix.h"
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <memory>

std::shared_ptr<Matrix> lpX = std::make_shared<Matrix>(12, 13);

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

//std::shared_ptr<Matrix> LossFcts::getDinput() const{return dinputs;}

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

    /*std::cout <<  "correct_confidences" << std::endl ;
    for (int j = 0; j < 5; j++) { 
        std::cout << correct_confidences[j] << " " ; 
    }
    std::cout << std::endl ;*/
    std::vector<double> nll = negative_log_likelihood(correct_confidences);

    /*std::cout <<  "nll" << std::endl ;
    for (int j = 0; j < 5; j++) { 
        std::cout << nll[j] << " " ; 
    }
    std::cout << std::endl ;*/

    return mean_vector(nll);
}


//Cross entropy loss function backward method
std::shared_ptr<Matrix> LossFcts::crossEntropyLoss_backward_softmax(std::shared_ptr<Matrix> dvalues, std::shared_ptr<Matrix> targets) {
    //std::vector<double> correct_confidences;
    std::shared_ptr<Matrix> dinputs = dvalues->copy();
    if(targets->getValues().size() == 1){
        for (int j = 0; j < targets->getNumCols(); j++) { 
            dinputs->setValue(j,  targets->getValue(0,j), dinputs->getValue(j, targets->getValue(0,j))-1);
        } 
    }

    return dinputs->divide(dvalues->getNumRows());
}

