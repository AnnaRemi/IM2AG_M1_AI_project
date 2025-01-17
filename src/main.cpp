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
        std::cerr << "Could not open the file" << std::endl;
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
    std::vector<std::vector<double>> weights1 = readCSV<double>("../data/wights_data_Layer1.csv");
    std::vector<std::vector<double>> weights2 = readCSV<double>("../data/wights_data_Layer2.csv");

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
    GradientDescent optimizer_GD(0.1);

    // First Layer
    std::shared_ptr<Layer> L1 = std::make_shared<Layer>(2, 64, weights1);

    // Second Layer
    std::shared_ptr<Layer> L2 = std::make_shared<Layer>(64, 3, weights2);

    // Activation functions
    ActivationFcts act1, act2;

    // Loss function
    LossFcts loss_fct;

    if (mode == "train")
    {
        int epochs = 100;
        // Train loop
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            std::cout << "step " << epoch + 1 << std::endl;

            L1->forward(spiral_data);
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

            if (epoch % 10 == 0)
            {
                std::cout << "Loss: " << loss << std::endl;
                std::cout << "Accuracy: " << acc << std::endl;
            }

            // Backward pass

            // Loss cross-entropy with softmax
            std::shared_ptr<Matrix> dinputs_loss = loss_fct.crossEntropyLoss_backward_softmax(output_softmax, y);

            // Second layer with softmax dinput
            L2->backward(dinputs_loss->getValues());
            std::vector<std::vector<double>> dinputs_L2 = L2->getDinputs();
            std::shared_ptr<Matrix> mat_dinp_L2 = std::make_shared<Matrix>(dinputs_L2.size(), dinputs_L2[0].size(), dinputs_L2);

            // Relu with second layer dinput
            std::shared_ptr<Matrix> dinputs_relu = act1.reluDerivative(mat_output_L1, mat_dinp_L2);

            // First layer with Relu dinput
            L1->backward(dinputs_relu->getValues());
            std::vector<std::vector<double>> dinputs_L1 = L1->getDinputs();
            std::shared_ptr<Matrix> mat_dinp_L1 = std::make_shared<Matrix>(dinputs_L1.size(), dinputs_L1[0].size(), dinputs_L1);

            // Update weights and biases :
            optimizer_GD.update_params(L1);
            optimizer_GD.update_params(L2);
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
