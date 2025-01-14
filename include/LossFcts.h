#ifndef LOSSFCTS_H
#define LOSSFCTS_H

#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>

#include "Matrix.h"

class LossFcts {
   /* private:
        std::shared_ptr<Matrix> dinputs;*/
    public:
        //std::shared_ptr<Matrix> getDinput() const;
        double crossEntropyLoss_forward(std::shared_ptr<Matrix>, std::shared_ptr<Matrix> );
        std::shared_ptr<Matrix> crossEntropyLoss_backward_softmax(std::shared_ptr<Matrix>, std::shared_ptr<Matrix> );
};

#endif // LOSSFCTS_H


