#ifndef ACTIVATIONFCTS_H
#define ACTIVATIONFCTS_H

#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>

#include "Matrix.h"

class ActivationFcts {
    public:
        std::vector<double> ReLU(const std::vector<double>& inputs);
        Matrix ReLU(Matrix );
        Matrix reluDerivative(Matrix dvalues, Matrix inputs); 
        Matrix Softmax(Matrix );
};

#endif // ACTIVATIONFCTS_H
