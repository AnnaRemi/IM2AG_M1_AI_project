
#include "ActivationFcts.h"
#include "Matrix.h"
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>



std::vector<std::vector<double>> maxOfEachRow(const std::vector<std::vector<double>> &matrix) {
   std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));
   for (long unsigned int i = 0; i < matrix.size(); ++i) {
      double maxVal = *std::max_element(matrix[i].begin(), matrix[i].end());
      std::fill(result[i].begin(), result[i].end(), maxVal);
   }
   return result;
}

// ReLU function
std::vector<double> ActivationFcts::ReLU(const std::vector<double>& inputs) {
   std::vector<double> result(inputs.size(), 0);
   for (long unsigned int i = 0; i < result.size(); ++i) {
      result[i] = std::max(0.0,inputs[i]);
   }
   return result;
}

Matrix ActivationFcts::ReLU(Matrix input){
   int r = input.getNumRows();
   int c = input.getNumCols();
   Matrix result = Matrix(r, c);
   
   for (int i = 0; i < r; ++i) {
      std::vector<double> row_relu = ReLU(input.getValues()[i]);
      result.setRowValues(i,row_relu);
   }

   return result;

}

// Derivative of the ReLU function
Matrix ActivationFcts::reluDerivative(Matrix dvalues, Matrix inputs) {
   Matrix dinputs = dvalues.copy();
   int r = dvalues.getNumRows();
   int c = dvalues.getNumCols();

   for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
         dinputs.setValue(i,j, inputs.getValue(i, j) > 0 ? 1.0 : 0.0);
      }
   }
   return dinputs;
}

// Softmax function
Matrix ActivationFcts::Softmax(Matrix input) {
   int r = input.getNumRows();
   int c = input.getNumCols();

   
   auto maxVals = maxOfEachRow(input.getValues());
   Matrix maxVals_mat = Matrix(r, c, maxVals);

   Matrix subts_mat = input - maxVals_mat;

   
   std::vector<std::vector<double>> expVals(r, std::vector<double>(c, 0.0));
   //double sum = 0.0;


   for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
         expVals[i][j] = std::exp(subts_mat.getValue(i,j));
      }
   }

   Matrix expMat = Matrix(r, c, expVals);
   std::vector<std::vector<double>> sumRows = expMat.sumRowsMat();
   Matrix sumMat = Matrix(r, c, sumRows);
   Matrix result = expMat/sumMat;
   

   return result;
  
}

// Derivative of the softmax function
/*double softmaxDerivative(double x) {
   return x * (1 - x);
}*/
