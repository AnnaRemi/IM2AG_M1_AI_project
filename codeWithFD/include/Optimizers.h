#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <vector>
#include <math.h>
#include <iostream>
#include <cstdlib>


#include "Matrix.h"
#include "Layer.h"


class Optimizers {
    public:
        virtual void update_params(std::shared_ptr<Layer>) = 0;
};

class GradientDescent: public Optimizers{
    private:
        double learning_rate;
    public:
        GradientDescent(double);
        void update_params(std::shared_ptr<Layer>);
};

#endif // OPTIMIZERS_H