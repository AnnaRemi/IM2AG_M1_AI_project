#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>

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

template<typename T>
void writeToFile( std::vector<std::vector<T>> matrix, std::string file_name){
    std::ofstream file(file_name);
    if (file.is_open()){
        for(const auto& row : matrix ){
            for( size_t i = 0; i < row.size(); i++){
                file << row[i];
                if(i < row.size()-1){
                    file << ",";
                }
            }
            file << "\n";
        }
        file.close();
    }
    else{
        std::cerr << "Could not open the file" << std::endl;
    }
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

    //Optimizer
    Optimizer optimizer_SGD;

    //std::vector<double> biases(spiral_data.size(), 0.0);
    // First Layer
    Layer L1(2,3);
    // ReLU
    ActivationFcts act1;

    // Second Layer
    Layer L2(3, 3);//Problem when the ni of L2 != nn of L2

    // Softmax
    ActivationFcts act2;

    // Loss
    LossFcts loss_fct;
    Matrix y(targets.size(),targets[0].size(),targets);
    y_transpose = y.transpose();

    int epochs = 1;

    
    //std::vector<std::vector<double>> losses_matrix(1, std::vector<double>(epochs, 0.0));
   std::vector<std::vector<double>> losses_matrix;
   std::vector<std::vector<double>> accuracies_matrix;
   std::vector<std::vector<int>> predictions_matrix;
    
    //Train loop
    for (int epoch = 0; epoch < epochs;epoch++){
        std::cout << "step " << epoch << std::endl;

        //std::vector<std::vector<double>> L1.get_weights();
        
        // Forward pass for the First Layer
        L1.forward(spiral_data);
        output_L1 = L1.getOutput();
        //std::cout <<  "L1 output " << output_L1.getNumRows() << " " << output_L1.getNumCols() << std::endl ;
        // output_L1.print();
        // ReLU activation
        output_relu = act1.ReLU(output_L1);
        //std::cout <<  "output_relu " << output_relu.getNumRows() << " " << output_relu.getNumCols() << std::endl ;
        //output_relu.print();

        // Forward pass for the Second Layer
        L2.forward(output_relu.getValues());
        output_L2 = L2.getOutput();
        //std::cout <<  "output_L2 " << output_L2.getNumRows() << " " << output_L2.getNumCols() << std::endl ;
        //output_L2.print();

        //Softmax activation
        output_softmax = act2.Softmax(output_L2);

        double loss = loss_fct.crossEntropyLoss_forward(output_softmax, y_transpose);
        std::cout <<  "Loss: " << loss << std::endl ;
        losses_matrix.push_back({loss});
        
        
        //std::cout <<  "output_softmax " << output_softmax.getNumRows() << " " << output_softmax.getNumCols() << std::endl ;
        //output_softmax.print();

        //Accuracy
        std::vector<int> predictions = output_softmax.argmaxRow();
        std::vector<int> true_values = y_transpose.argmaxRow();
        std::vector<int> accuracy;
        for (long unsigned int j = 0; j < predictions.size(); j++) { 
             accuracy.push_back(predictions[j] - true_values[j]); 
         }
        double acc = mean_vector(accuracy);
        std::cout <<  "Accuracy: " << acc << std::endl ;
        accuracies_matrix.push_back({acc});
        predictions_matrix.push_back(predictions);


        //Backward pass :
        //<=>lossActivation.backward(loss_activation.output, y) in python
        dinputs_loss = loss_fct.crossEntropyLoss_backward_softmax(output_softmax, y);
        
        //<=>dense2.backward(loss_activation.dinputs) in python
        L2.backward(dinputs_loss.getValues());
        std::cout <<  " L2 dinputs numRows " <<  L2.get_dinputs().getNumRows() << std::endl;
        std::cout <<  " L2 dinputs numCols " <<  L2.get_dinputs().getNumCols() << std::endl ;
        L2.get_dinputs().print();

        //<=>activation1.backward(dense2.dinputs) in python
        dinputs_ReLU = act1.reluDerivative(L2.get_dinputs(), output_L1);
        //=>dense1.backward(activation1.dinputs) in python
        std::cout << "here" << std::endl;

        L1.backward(dinputs_ReLU.getValues());
        //Update weights and biases :
        optimizer_SGD.update_parameters(L1);
        optimizer_SGD.update_parameters(L2);
        
    } 

    //Writting results to file
    writeToFile(losses_matrix,"losses.csv");
    writeToFile(predictions_matrix, "predictions.csv");
    writeToFile(accuracies_matrix,"accuracies.csv");

    

    return 0;
}
