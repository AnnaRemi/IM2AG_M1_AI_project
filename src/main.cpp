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


template <typename T>
double mean_vector(const std::vector<T> &vec)
{
    T sum = 0;
    for (int i = 0; i < vec.size(); ++i)
    {
        sum += vec[i];
    }

    return static_cast<double>(sum) / vec.size();
}

template <typename T>
void writeToFile(std::vector<std::vector<T>> matrix, std::string file_name)
{
    std::ofstream file(file_name);
    if (file.is_open())
    {
        for (const auto &row : matrix)
        {
            for (size_t i = 0; i < row.size(); i++)
            {
                file << row[i];
                if (i < row.size() - 1)
                {
                    file << ",";
                }
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cerr << "Could not open the file" << std::endl;
    }
}

template <typename T>
std::vector<std::vector<T>> readCSV(const std::string &file_name)
{
    std::vector<std::vector<T>> data;
    std::ifstream file(file_name);
    std::string line, cell;
    if (file.is_open())
    {
        while (std::getline(file, line))
        {
            std::vector<T> row;
            std::stringstream lineStream(line);
            while (std::getline(lineStream, cell, ','))
            {
                row.push_back(std::stod(cell));
            }
            data.push_back(row);
        }
        file.close();
    }
    else
    {
        std::cerr << "Could not open the file " << file_name << std::endl;
    }
    return data;
}

std::vector<double> readPointFromConsole()
{
    std::vector<double> point(2);
    std::cout << "Enter coordinate 1: ";
    std::cin >> point[0];
    std::cout << "Enter coordinate 2: ";
    std::cin >> point[1];
    return point;
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <mode>\n";
        std::cerr << "Modes: train, test\n";
        return 1;
    }
    std::string mode = argv[1];

    std::cout << "Training mode selected.\n";

    // Reading data
    std::ifstream f("../data/spiral_data.json");

    if (!f.is_open())
    {
        std::cerr << "Could not open the JSON file";
        return 1;
    }

    json data = json::parse(f);

    f.close();

    std::vector<std::vector<double>> spiral_data;
    std::vector<std::vector<double>> targets;
    Matrix mat_weights1 = Matrix(2,64);
    
    std::vector<std::vector<double>> weights1 = readCSV<double>("../data/wights_data_Layer1.csv");//mat_weights1.getValues();
    Matrix mat_weights2 = Matrix(64,3);
   
    std::vector<std::vector<double>> weights2 = readCSV<double>("../data/wights_data_Layer2.csv");// mat_weights2.getValues();

    for (const auto &item : data["X"])
    {
        std::vector<double> row;
        for (const auto &value : item)
        {
            row.push_back(value.get<double>());
        }
        spiral_data.push_back(row);
    }

    for (const auto &item : data["y"])
    {
        std::vector<double> row;
        for (const auto &value : item)
        {
            row.push_back(value.get<double>());
        }
        targets.push_back(row);
    }

    std::shared_ptr<Matrix> y = std::make_shared<Matrix>(targets.size(), targets[0].size(), targets);
    std::shared_ptr<Matrix> y_transpose = y->transpose();

    std::vector<std::vector<double>> losses_matrix;
    std::vector<std::vector<double>> accuracies_matrix;
    std::vector<std::vector<double>> predictions_matrix;

    // Optimizer
    //GradientDescent optimizer_GD(0.5);
    GradientDescentWithDecay optimizer_SGD(1,1e-3);
    //Adam optimizer_Adam(0.02,1e-5,1e-7,0.9,0.999);
    //RandomUpdate random;

    // First Layer
    std::shared_ptr<Layer> L1 = std::make_shared<Layer>(2, 64, weights1);
    
    /*std::cout << "L1->getDbiases() at beginning" << std::endl;
    std::cout <<  "[ " ;
    for (int i = 0; i < L1->getDbiases().size(); i++) {
        std::cout <<  "[ " ;
        for (int j = 0; j < L1->getDbiases()[0].size(); j++) { 
            std::cout << L1->getDbiases()[i][j] << " " ; 
        }
        std::cout <<  " ]" << std::endl ;
    }
    
    std::cout <<  " ]" << std::endl ;
    std::cout <<  " end" << std::endl ;*/

    
    // Second Layer
    std::shared_ptr<Layer> L2 = std::make_shared<Layer>(64, 3, weights2);
    

    // Activation functions
    ActivationFcts act1, act2;

    // Loss function
    LossFcts loss_fct;

    if (mode == "train")
    {
        int epochs = 10000;
        /*int lowest_loss = 9999999;
        std::vector<std::vector<double>> best_weights_1 = L1->getWeights();
        std::vector<std::vector<double>> best_weights_2 = L2->getWeights();
        std::vector<std::vector<double>> best_biases_1 = L1->getBiases();
        std::vector<std::vector<double>> best_biases_2 = L2->getBiases();*/
        // Train loop
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            
            //random.update_params(L1);
            //random.update_params(L2);
            
            L1->forward(spiral_data);
            std::vector<std::vector<double>> output_L1 = L1->getOutput();
            std::shared_ptr<Matrix> mat_output_L1 = std::make_shared<Matrix>(output_L1.size(), output_L1[0].size(), output_L1);
            //std::cout << "mat_output_L1 " << mat_output_L1->getNumRows() << " "  << mat_output_L1->getNumCols() << std::endl;
            /*std::cout << "mat_output_L1" << std::endl;
            std::cout <<  "[ " ;
            for (int i = 295; i < mat_output_L1->getNumRows(); i++) {
                std::cout <<  "[ " ;
                for (int j = 0; j < mat_output_L1->getNumCols(); j++) { 
                    std::cout << mat_output_L1->getValue(i,j) << " " ; 
                }
                std::cout <<  " ]" << std::endl ;
            }
            
            std::cout <<  " ]" << std::endl ;
            std::cout <<  " end" << std::endl ;*/

            // ReLU
            std::shared_ptr<Matrix> output_relu = act1.ReLU(mat_output_L1);

            /*std::cout << "output_relu" << std::endl;
            std::cout <<  "[ " ;
            for (int i = 295; i < output_relu->getNumRows(); i++) {
                std::cout <<  "[ " ;
                for (int j = 0; j < output_relu->getNumCols(); j++) { 
                    std::cout << output_relu->getValue(i,j) << " " ; 
                }
                std::cout <<  " ]" << std::endl ;
            }
            
            std::cout <<  " ]" << std::endl ;
            std::cout <<  " end" << std::endl ;*/
            

            // Second Layer
            L2->forward(output_relu->getValues());
            std::vector<std::vector<double>> output_L2 = L2->getOutput();
            std::shared_ptr<Matrix> mat_output_L2 = std::make_shared<Matrix>(output_L2.size(), output_L2[0].size(), output_L2);

            /*std::cout << "mat_output_L2" << std::endl;
            std::cout <<  "[ " ;
            for (int i = 295; i < mat_output_L2->getNumRows(); i++) {
                std::cout <<  "[ " ;
                for (int j = 0; j < mat_output_L2->getNumCols(); j++) { 
                    std::cout << mat_output_L2->getValue(i,j) << " " ; 
                }
                std::cout <<  " ]" << std::endl ;
            }
            
            std::cout <<  " ]" << std::endl;
            std::cout <<  " end" << std::endl ;*/
            

            // Softmax
            std::shared_ptr<Matrix> output_softmax = act2.Softmax(mat_output_L2);
           //std::cout << "output_softmax " << output_softmax->getNumRows() << " "  << output_softmax->getNumCols() << std::endl;
            

            // Loss
            double loss = loss_fct.crossEntropyLoss_forward(output_softmax, y_transpose);
            losses_matrix.push_back({loss});

            // Accuracy
            std::vector<double> predictions = output_softmax->argmaxRow();
            std::vector<double> true_values;
            
            if (y_transpose->getNumRows() > 1)
            {
                true_values = y_transpose->argmaxRow();
            }
            else
            {
                true_values = y_transpose->getValues()[0];
            }
            std::vector<int> accuracy(predictions.size(), 0);
            int sum = 0;
            for (int j = 0; j < predictions.size(); j++)
            {
                if (predictions[j] == true_values[j])
                {
                    accuracy[j] = 1;
                    sum++;
                }
            }

            double acc = mean_vector(accuracy);
            accuracies_matrix.push_back({acc});
            predictions_matrix.push_back(predictions);

            if (epoch % 100 == 0)
            {
                std::cout << "epoch " << epoch  << std::endl;
                std::cout << "Loss: " << loss << std::endl;
                std::cout << "Accuracy: " << acc << std::endl;
                //std::cout << "lr: " << optimizer_Adam.getCurrent_learning_rate() << std::endl;
            }
            
            /*if (loss < lowest_loss){
                std::cout << "epoch " << epoch + 1 << std::endl;
                std::cout << "Loss: " << loss << std::endl;
                std::cout << "Accuracy: " << acc << std::endl;
                lowest_loss = loss;
                best_weights_1 = L1->getWeights();
                best_weights_2 = L2->getWeights();
                best_biases_1 = L1->getBiases();
                best_biases_2 = L2->getBiases();
            }
            else{
                lowest_loss = loss;
                L1->setWeights(best_weights_1);
                L2->setWeights(best_weights_2);
                L1->setBiases(best_biases_1);
                L2->setBiases(best_biases_2);
            }*/
            // Backward pass

            //Loss cross-entropy with softmax

            /*std::cout << "output_softmax" << std::endl;
            std::cout <<  "[ " ;
            for (int i = 295; i < output_softmax->getNumRows(); i++) {
                std::cout <<  "[ " ;
                for (int j = 0; j < output_softmax->getNumCols(); j++) { 
                    std::cout << output_softmax->getValue(i,j) << " " ; 
                }
                std::cout <<  " ]" << std::endl ;
            }
            
            std::cout <<  " ]" << std::endl;
            std::cout <<  " end" << std::endl ;*/

            std::shared_ptr<Matrix> dinputs_loss = loss_fct.crossEntropyLoss_backward_softmax(output_softmax, y);
            //std::cout << "dinputs_loss " << dinputs_loss->getNumRows() << " "  << dinputs_loss->getNumCols() << std::endl;
           
            /*std::cout << "dinputs_loss" << std::endl;
            std::cout <<  "[ " ;
            for (int i = 295; i < dinputs_loss->getNumRows(); i++) {
                std::cout <<  "[ " ;
                for (int j = 0; j < dinputs_loss->getNumCols(); j++) { 
                    std::cout << dinputs_loss->getValue(i,j) << " " ; 
                }
                std::cout <<  " ]" << std::endl ;
            }
            
            std::cout <<  " ]" << std::endl ;
            std::cout <<  " end" << std::endl ;*/

            // Second layer with softmax dinput
            L2->backward(dinputs_loss->getValues());
            std::vector<std::vector<double>> dinputs_L2 = L2->getDinputs();
            std::shared_ptr<Matrix> mat_dinp_L2 = std::make_shared<Matrix>(dinputs_L2.size(), dinputs_L2[0].size(), dinputs_L2);
            //std::cout << "mat_dinp_L2 " << mat_dinp_L2->getNumRows() << " "  << mat_dinp_L2->getNumCols() << std::endl;
            /*std::cout << "mat_dinp_L2" << std::endl;
            std::cout <<  "[ " ;
            for (int i = 295; i < mat_dinp_L2->getNumRows(); i++) {
                std::cout <<  "[ " ;
                for (int j = 0; j < mat_dinp_L2->getNumCols(); j++) { 
                    std::cout << mat_dinp_L2->getValue(i,j) << " " ; 
                }
                std::cout <<  " ]" << std::endl ;
            }
            
            std::cout <<  " ]" << std::endl ;
            std::cout <<  " end" << std::endl ;*/
            

            // Relu with second layer dinput
            std::shared_ptr<Matrix> dinputs_relu = act1.reluDerivative(mat_output_L1, mat_dinp_L2);
            
            /*std::cout << "dinputs_relu" << std::endl;
            std::cout <<  "[ " ;
            for (int i = 295; i < dinputs_relu->getNumRows(); i++) {
                std::cout <<  "[ " ;
                for (int j = 0; j < dinputs_relu->getNumCols(); j++) { 
                    std::cout << dinputs_relu->getValue(i,j) << " " ; 
                }
                std::cout <<  " ]" << std::endl ;
            }
            
            std::cout <<  " ]" << std::endl ;
            std::cout <<  " end" << std::endl ;*/

            // First layer with Relu dinput
            L1->backward(dinputs_relu->getValues());
            std::vector<std::vector<double>> dinputs_L1 = L1->getDinputs();
            std::shared_ptr<Matrix> mat_dinp_L1 = std::make_shared<Matrix>(dinputs_L1.size(), dinputs_L1[0].size(), dinputs_L1);

            /*std::cout << "mat_dinp_L1" << std::endl;
            std::cout <<  "[ " ;
            for (int i = 295; i < mat_dinp_L1->getNumRows(); i++) {
                std::cout <<  "[ " ;
                for (int j = 0; j < mat_dinp_L1->getNumCols(); j++) { 
                    std::cout << mat_dinp_L1->getValue(i,j) << " " ; 
                }
                std::cout <<  " ]" << std::endl ;
            }
            
            std::cout <<  " ]" << std::endl ;
            std::cout <<  " end" << std::endl ;*/

            // Update weights and biases with GD:
            //optimizer_GD.update_params(L1);
            //optimizer_GD.update_params(L2);
            //random.update_params(L1);
            //random.update_params(L2);

            /*optimizer_Adam.pre_update_params();
            optimizer_Adam.update_params(L1, 1);
            optimizer_Adam.update_params(L2,2);
            optimizer_Adam.post_update_params();*/
            
            optimizer_SGD.pre_update_params();
            optimizer_SGD.update_params(L1,1);
            optimizer_SGD.update_params(L2,2);
            optimizer_SGD.post_update_params();

            /*std::cout << "L1->getWeights()" << std::endl;
            std::cout <<  "[ " ;
            for (int i = 0; i < L1->getWeights().size(); i++) {
                std::cout <<  "[ " ;
                for (int j = 0; j < L1->getWeights()[0].size(); j++) { 
                    std::cout << L1->getWeights()[i][j] << " " ; 
                }
                std::cout <<  " ]" << std::endl ;
            }
            
            std::cout <<  " ]" << std::endl ;
            std::cout <<  " end" << std::endl ;*/

            /*std::cout << "L1->getBiases()" << std::endl;
            std::cout <<  "[ " ;
            for (int i = 0; i < L1->getBiases().size(); i++) {
                std::cout <<  "[ " ;
                for (int j = 0; j < L1->getBiases()[0].size(); j++) { 
                    std::cout << L1->getBiases()[i][j] << " " ; 
                }
                std::cout <<  " ]" << std::endl ;
            }
            
            std::cout <<  " ]" << std::endl;
            std::cout <<  " end" << std::endl ;*/
            
        }

        // Writting results to file
        writeToFile(losses_matrix, "losses.csv");
        writeToFile(predictions_matrix, "predictions.csv");
        writeToFile(accuracies_matrix, "accuracies.csv");
    }
    else if (mode == "test")
    {
        std::cout << "Testing mode selected.\n";
        // Read point from console
        std::vector<double> point = readPointFromConsole();

        // Predict label
        std::vector<std::vector<double>> input = {{point}};
        std::shared_ptr<Matrix> input_matrix = std::make_shared<Matrix>(input.size(), input[0].size(), input);
        L1->forward(input);
        std::vector<std::vector<double>> output_L1 = L1->getOutput();
        std::shared_ptr<Matrix> mat_output_L1 = std::make_shared<Matrix>(output_L1.size(), output_L1[0].size(), output_L1);

        // ReLU
        std::shared_ptr<Matrix> output_relu = act1.ReLU(mat_output_L1);

        // Second Layer
        L2->forward(output_relu->getValues());
        std::vector<std::vector<double>> output_L2 = L2->getOutput();
        std::shared_ptr<Matrix> mat_output_L2 = std::make_shared<Matrix>(output_L2.size(), output_L2[0].size(), output_L2);

        // Softmax
        std::shared_ptr<Matrix> output_softmax = act2.Softmax(mat_output_L2);

        std::vector<double> probabilities = output_softmax->getValues()[0];
        int predicted_label = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));

        std::cout << "Predicted label: " << predicted_label << std::endl;
    }
    else
    {
        std::cerr << "Invalid mode: " << mode << "\n";
        std::cerr << "Modes: train, test\n";
        return 1;
    }

    return 0;
}
