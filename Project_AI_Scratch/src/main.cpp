#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <memory>

#include "../include/Layer.h"
#include "../include/Matrix.h"
#include "../include/ActivationFcts.h"
#include "../include/LossFcts.h"
#include "../include/Optimizers.h"



#include <fstream>
#include "../include/json.hpp"
using json = nlohmann::json;

template<typename T>
double mean_vector(const std::vector<T>& vec){
    T sum = 0;
    for (int i = 0; i < vec.size(); ++i) {
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

//std::shared_ptr<Matrix> lpX = std::make_shared<Matrix>(12, 13);

int main(){

    //Reading spiral data
    std::ifstream f("../data/vertical_data.json");
    
    if(!f.is_open()){
        std::cerr << "Could not open the JSON file";
        return 1;
    }
    
    json data = json::parse(f);

    f.close();

    std::vector<std::vector<double>> vertical_data;
    std::vector<std::vector<double>> targets;

    for (const auto& item : data["X"]){
        std::vector<double> row;
        for (const auto& value : item){
            row.push_back(value.get<double>());
        }
        vertical_data.push_back(row);
    }

    for (const auto& item : data["y"]){
        std::vector<double> row;
        for (const auto& value : item){
            row.push_back(value.get<double>());
        }
        targets.push_back(row);
    }
    
    std::shared_ptr<Matrix> y = std::make_shared<Matrix>(targets.size(),targets[0].size(),targets);
    std::shared_ptr<Matrix> y_transpose = y->transpose();

    std::vector<std::vector<double>> losses_matrix;
    std::vector<std::vector<double>> accuracies_matrix;
    std::vector<std::vector<int>> predictions_matrix;

    //Optimizer
    GradientDescent optimizer_GD(0.001);

    //First Layer
    Layer L1(2,64);
    //Second Layer
    Layer L2(64,3);
    ActivationFcts act1, act2;
    LossFcts loss_fct;

    int epochs = 10;
    //Train loop
    for (int epoch = 0; epoch < epochs;epoch++){
        std::cout << "step " << epoch+1 << std::endl;

        L1.forward(vertical_data);
        std::vector<std::vector<double>> output_L1 = L1.getOutput();
        std::shared_ptr<Matrix> mat_output_L1 = std::make_shared<Matrix>(output_L1.size(), output_L1[0].size(), output_L1);
        //mat_output_L1->print();

        //ReLU
        std::shared_ptr<Matrix> output_relu = act1.ReLU(mat_output_L1);
        //std::cout <<  "output_relu " << output_relu->getNumRows() << " " << output_relu->getNumCols() << std::endl ;
        //output_relu->print();

        //Second Layer
        L2.forward(output_relu->getValues());
        std::vector<std::vector<double>> output_L2 = L2.getOutput();
        std::shared_ptr<Matrix> mat_output_L2 = std::make_shared<Matrix>(output_L2.size(), output_L2[0].size(), output_L2);

        
        //std::cout <<  "output_L2 " << output_L2->getNumRows() << " " << output_L2->getNumCols() << std::endl ;
        //output_L2->print();
        
        //Softmax 
        std::shared_ptr<Matrix> output_softmax = act2.Softmax(mat_output_L2);
        
        //std::cout <<  "output_softmax " << output_softmax->getNumRows() << " " << output_softmax->getNumCols() << std::endl ;
        //output_softmax->print();

        /*std::vector<double> probabilities = output_softmax->sumMat();
        std::cout <<  "Probabilities" << std::endl ;
        for (int j = 0; j < probabilities.size(); j++) { 
            std::cout << probabilities[j] << " " ; 
        
        std::cout << std::endl;*/

        
        //Loss
        
        double loss = loss_fct.crossEntropyLoss_forward(output_softmax, y_transpose);
        std::cout <<  "Loss: " << loss << std::endl ;
        losses_matrix.push_back({loss});
        
        //Accuracy
        std::vector<int> predictions = output_softmax->argmaxRow();
        std::vector<int> true_values = y_transpose->argmaxRow();
        

        std::vector<int> accuracy;
        for (int j = 0; j < predictions.size(); j++) { 
            accuracy.push_back(predictions[j] - true_values[j]); 
        }
        
        double acc = mean_vector(accuracy);
        std::cout <<  "Accuracy: " << acc << std::endl ;
        accuracies_matrix.push_back({acc});
        predictions_matrix.push_back(predictions);


        //std::cout << "Finished forward" << std::endl;
        //Backward pass
    

        //Loss cross-entropy with softmax
        std::shared_ptr<Matrix> dinputs_loss = loss_fct.crossEntropyLoss_backward_softmax(output_softmax, y);
        
        //std::cout << "Before L2 backward" << std::endl;
        //Second layer with softmax dinput
        L2.backward(dinputs_loss->getValues());
        std::vector<std::vector<double>> dinputs_L2 = L2.getDinputs();
        std::shared_ptr<Matrix> mat_dinp_L2 = std::make_shared<Matrix>(dinputs_L2.size(), dinputs_L2[0].size(), dinputs_L2);
        //std::cout << "Finished L2 backward" << std::endl;

        //Relu with second layer dinput
        std::shared_ptr<Matrix> dinputs_relu = act1.reluDerivative(mat_output_L1, mat_dinp_L2);

        
        

        //First layer with Relu dinput
        L1.backward(dinputs_relu->getValues());
        std::vector<std::vector<double>> dinputs_L1 = L1.getDinputs();
        std::shared_ptr<Matrix> mat_dinp_L1 = std::make_shared<Matrix>(dinputs_L1.size(), dinputs_L1[0].size(), dinputs_L1);
        //std::cout <<  " dinputs_L1 numRows " <<  mat_dinp_L1->getNumRows() << std::endl;
        //std::cout <<  " dinputs_L1 numCols " <<  mat_dinp_L1->getNumCols() << std::endl ;
        //mat_dinp_L1->print();
        std::cout << "Finished backward" << std::endl;


        //Update weights and biases :
        optimizer_GD.update_params(L1);
        optimizer_GD.update_params(L2);

        
    }

    //Writting results to file
    writeToFile(losses_matrix,"losses.csv");
    writeToFile(predictions_matrix, "predictions.csv");
    writeToFile(accuracies_matrix,"accuracies.csv");
    
    return 0;
}

/*int main(){
    std::vector<std::vector<double>> i = {{1, 2, 3, 2.5},
        {2., 5., -1., 2},
        {-1.5, 2.7, 3.3, -0.8}};
    std::vector<std::vector<double>> w = {{0.2, 0.8, -0.5, 1},
        {0.5, -0.91, 0.26, -0.5},
        {-0.26, -0.27, 0.17, 0.87}};
    std::vector<double> b = {{2, 3, 0.5}};

    std::shared_ptr<Matrix> inputs = std::make_shared<Matrix>(4, 3, i);
    std::shared_ptr<Matrix> weights = std::make_shared<Matrix>(4, 3, w);
    std::shared_ptr<Matrix> biases = std::make_shared<Matrix>(1, 3, b);



    return 0;
}*/