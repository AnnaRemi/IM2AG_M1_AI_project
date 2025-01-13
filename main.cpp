#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "Neuron.h"
#include "Layer.h"
#include "Matrix.h"
#include "ActivationFcts.h"
#include "LossFcts.h"



#include <fstream>
#include "json.hpp"
using json = nlohmann::json;

template<typename T>
double mean_vector(const std::vector<T>& vec){
    T sum = 0;
    for (long unsigned int i = 0; i < vec.size(); ++i) {
        sum += vec[i];
    }
    return sum / vec.size();
}


int main(){
    //Reading spiral data
    std::ifstream f("spiral_data.json");
    
    if(!f.is_open()){
        std::cerr << "Could not open the JSON file\n";
        return 1;
    }
    
    json data = json::parse(f);

    f.close();

    std::vector<std::vector<double>> spiral_data;
    std::vector<std::vector<double>> targets;

    for (const auto& item : data["X"]){
        std::vector<double> row;
        for (const auto& value : item){
            row.push_back(value.get<double>());
        }
        spiral_data.push_back(row);
    }

    for (const auto& item : data["y"]){
        std::vector<double> row;
        for (const auto& value : item){
            row.push_back(value.get<double>());
        }
        targets.push_back(row);
    }
    Matrix input_spirale(spiral_data.size(), spiral_data[0].size(), spiral_data);

    // we declare the Matrices that we will use in the train loop
    Matrix output_L1, output_relu, output_L2, output_softmax, y_transpose, dinputs_loss, dinputs_ReLU;
    double loss;

    //Optimizer
    Optimizer optimizer_SGD(.001);

    // First Layer
    Layer L1(2,64);
    // ReLU
    ActivationFcts act1;

    // Second Layer
    Layer L2(64, 3);//Problem when the ni of L2 != nn of L2

    // Softmax
    ActivationFcts act2;

    // Loss
    LossFcts loss_fct;
    Matrix y(targets.size(),targets[0].size(),targets);
    y_transpose = y.transpose();


    //Train loop
    for (int epoch = 0; epoch < 1000;epoch++){
        // Forward pass for the First Layer

        // L1.print_weights();
        L1.forward(spiral_data);

        // L1.print_weights();
        output_L1 = L1.getOutput();

        // ReLU activation
        output_relu = act1.ReLU(output_L1);

        // Forward pass for the Second Layer
        L2.forward(output_relu.getValues());

        output_L2 = L2.getOutput();

        //Softmax activation
        output_softmax = act2.Softmax(output_L2);
        if(epoch%100 == 0){
            std::cout << "step " << epoch << std::endl;
            loss = loss_fct.crossEntropyLoss_forward(output_softmax, y_transpose);
            std::cout <<  "Loss: " << loss << std::endl ;
        }

        //Accuracy
        // std::vector<int> predictions = output_softmax.argmaxRow();
        // std::vector<int> true_values = y_transpose.argmaxRow();
        // std::vector<int> accuracy;
        // for (long unsigned int j = 0; j < predictions.size(); j++) { 
        //     accuracy.push_back(predictions[j] - true_values[j]); 
        // }
        // double acc = mean_vector(accuracy);
        // std::cout <<  "Accuracy: " << acc << std::endl ;

        //Backward pass :
        //<=>lossActivation.backward(loss_activation.output, y) in python
        dinputs_loss = loss_fct.crossEntropyLoss_backward_softmax(output_softmax, y);
        //<=>dense2.backward(loss_activation.dinputs) in python
        L2.backward(dinputs_loss.getValues());

        //<=>activation1.backward(dense2.dinputs) in python
        dinputs_ReLU = act1.reluDerivative(L2.get_dinputs(), output_L1);

        //=>dense1.backward(activation1.dinputs) in python
        L1.backward(dinputs_ReLU.getValues());

        //Update weights and biases :
        optimizer_SGD.update_parameters(L1);
        optimizer_SGD.update_parameters(L2);
    }
    return 0;
}
