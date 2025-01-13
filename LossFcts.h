#ifndef LOSSFCTS_H
#define LOSSFCTS_H

#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>

#include "Matrix.h"

class LossFcts {
   /* private:
        Matrix* dinputs;*/
    public:
        //Matrix* getDinput() const;
        double crossEntropyLoss_forward(Matrix, Matrix );
        Matrix crossEntropyLoss_backward_softmax(Matrix, Matrix);
};

#endif // LOSSFCTS_H


