#ifndef ACTIVATIONFCTS_H
#define ACTIVATIONFCTS_H

#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>

#include "Matrix.h"

class ActivationFcts {
    public:
        std::vector<double> ReLU(const std::vector<double>& inputs);
        std::shared_ptr<Matrix> ReLU(std::shared_ptr<Matrix> );
        std::shared_ptr<Matrix> reluDerivative(std::shared_ptr<Matrix> dvalues, std::shared_ptr<Matrix> inputs);
        std::shared_ptr<Matrix> Softmax(std::shared_ptr<Matrix> );
};

#endif // ACTIVATIONFCTS_H