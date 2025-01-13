#include "LossFcts.h"
#include "Matrix.h"
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>

/*LossFcts::LossFcts(){
   dinputs = nullptr;
}*/

double clip(const double &value, const double &min, const double &max) {
   return std::max(min,std::min(value,max));
}

std::vector<double> clip(const std::vector<double> &values, const double &min, const double &max) {
   std::vector<double> result(values.size());

   std::transform(values.begin(), values.end(), result.begin(), [&min,&max](const double& val){return clip(val,min,max);});

   return result;
}

Matrix clip(const std::vector<std::vector<double>> &matrix, const double &min, const double &max) {
    std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));

    for (long unsigned int i = 0; i < matrix.size(); ++i) {
            auto part_result = clip(matrix[i], min, max);
            for (long unsigned int j = 0; j < matrix[0].size(); ++j) {
                result[i][j] = part_result[j];
            }
    }

    Matrix result_mat = Matrix(result.size(), result[0].size(),result);
    return result_mat;
}

std::vector<double> negative_log_likelihood(const std::vector<double>& vec){
    std::vector<double> result;
    for (long unsigned int i = 0; i < vec.size(); ++i) {
        result.push_back(-std::log(vec[i]));
    }
    return result;
}

double mean_vector(const std::vector<double>& vec){
    double sum = 0;
    for (long unsigned int i = 0; i < vec.size(); ++i) {
        sum += vec[i];
    }
    return sum / vec.size();
}

//Matrix* LossFcts::getDinput() const{return dinputs;}

//Cross entropy loss function forward method
double LossFcts::crossEntropyLoss_forward(Matrix predictions, Matrix targets) {
    std::vector<double> correct_confidences;
    Matrix clipped_prediction = clip(predictions.getValues(), 1e-7, 1 - 1e-7);
    if(targets.getValues().size() == 1){
        for (int j = 0; j < targets.getNumCols(); j++) { 
            correct_confidences.push_back(predictions.getValue(j, targets.getValue(0,j)));
        } 
    }
    else{
        Matrix mat_mult = clipped_prediction * targets;
        correct_confidences = mat_mult.sumMat();
    }
    std::vector<double> nll = negative_log_likelihood(correct_confidences);

    return mean_vector(nll);
}


//Cross entropy loss function backward method
Matrix LossFcts::crossEntropyLoss_backward_softmax(Matrix dvalues, Matrix targets) {
    Matrix dinputs = dvalues.copy();
    if(targets.getValues().size() == 1){
        for (int j = 0; j < targets.getNumCols(); j++) { 
            dinputs.setValue(j,  targets.getValue(0,j), dinputs.getValue(j, targets.getValue(0,j))-1);
        } 
    }

    return dinputs.divide(dvalues.getNumRows());
}

