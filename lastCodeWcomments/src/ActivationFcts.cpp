/**
 * @file ActivationFcts.cpp
 * @brief Contains the definitions of the functions declared in ActivationFcts.h
 */

#include "../include/ActivationFcts.h"
#include "../include/Matrix.h"
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <memory>




std::vector<std::vector<double>> maxOfEachRow(const std::vector<std::vector<double>> &matrix) {
   std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));

   for (int i = 0; i < matrix.size(); ++i) {
      double maxVal = *std::max_element(matrix[i].begin(), matrix[i].end());
      std::fill(result[i].begin(), result[i].end(), maxVal);
   }

   return result;
}


// ReLU function

std::vector<double> ActivationFcts::ReLU(const std::vector<double>& inputs) {
   std::vector<double> result(inputs.size(), 0);
   for (int i = 0; i < result.size(); ++i) {
      result[i] = std::max(0.0,inputs[i]);
   }
   return result;
}

std::shared_ptr<Matrix> ActivationFcts::ReLU(std::shared_ptr<Matrix> input){
   int r = input->getNumRows();
   int c = input->getNumCols();
   
   std::shared_ptr<Matrix> result = std::make_shared<Matrix>(r, c);
   
   for (int i = 0; i < r; ++i) {
      std::vector<double> row_relu = ReLU(input->getValues()[i]);
      result->setRowValues(i,row_relu);
   }

   return result;

}


// Derivative of the ReLU function

std::shared_ptr<Matrix> ActivationFcts::reluDerivative(std::shared_ptr<Matrix> dvalues, std::shared_ptr<Matrix> inputs) {
   std::shared_ptr<Matrix> dinputs = dvalues->copy();
   int r = inputs->getNumRows();
   int c = inputs->getNumCols();

   for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
         dinputs->setValue(i,j, dinputs->getValue(i, j) > 0 ? inputs->getValue(i, j): 0.0);
      }
   }
  
   return dinputs;
}


// Softmax function

std::shared_ptr<Matrix> ActivationFcts::Softmax(std::shared_ptr<Matrix> input) {
   int r = input->getNumRows();
   int c = input->getNumCols();
      
   std::vector<std::vector<double>> maxVals = maxOfEachRow(input->getValues());
   std::shared_ptr<Matrix> maxVals_mat = std::make_shared<Matrix>(maxVals.size(), maxVals[0].size(), maxVals);
   std::shared_ptr<Matrix> subts_mat = *input - *maxVals_mat;   
   std::vector<std::vector<double>> expVals(r, std::vector<double>(c, 0.0));
   double sum = 0.0;


   for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
         expVals[i][j] = std::exp(subts_mat->getValue(i,j));
      }
   }

   std::shared_ptr<Matrix> expMat = std::make_shared<Matrix>(r, c, expVals);
   std::vector<std::vector<double>> sumRows = expMat->sumRowsMat();
   std::shared_ptr<Matrix> sumMat = std::make_shared<Matrix>(r, c, sumRows);
   std::shared_ptr<Matrix> result = (*expMat)/(*sumMat);
   

   return result;
  
}

