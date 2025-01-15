#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <memory>
#include <random>

#include "../include/Layer.h"
#include "../include/Matrix.h"
#include "../include/ActivationFcts.h"
#include "../include/LossFcts.h"
#include "../include/Optimizers.h"
#include <fstream>
#include "../include/json.hpp"
using json = nlohmann::json;

/**
 * @brief Generates a random matrix with values between minVal and maxVal.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param minVal Minimum value for the random elements.
 * @param maxVal Maximum value for the random elements.
 * @return A matrix of dimensions rows x cols filled with random values.
 */
std::vector<std::vector<double>> RandomVec(int rows, int cols, double minVal = -1.0, double maxVal = 1.0)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(minVal, maxVal);
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = dist(gen);
    return matrix;
}

template <typename T>
void writeToFile(const std::vector<std::vector<T>> &matrix, const std::string &file_name)
{
    std::ofstream file(file_name);
    if (file.is_open())
    {
        for (const auto &row : matrix)
        {
            for (size_t i = 0; i < row.size(); i++)
            {
                file << row[i] << (i < row.size() - 1 ? "," : "");
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

/**
 * @brief Trains a neural network using randomly initialized weights and biases.
 * @param inputs Input data matrix.
 * @param neurons_L1 Number of neurons in the first layer.
 * @param neurons_L2 Number of neurons in the second layer.
 * @param epochs Number of training epochs.
 */
void trainNeuralNetworkRandom(std::vector<std::vector<double>> inputs, int neurons_L1, int neurons_L2, int epochs = 100)
{

    std::ifstream f("../data/vertical_data.json");
    if (!f.is_open())
    {
        std::cerr << "Could not open the JSON file";
        return;
    }
    json data = json::parse(f);
    f.close();

    std::vector<std::vector<double>> targets;
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
    std::vector<std::vector<int>> predictions_matrix;

    Layer L1(inputs[0].size(), neurons_L1);
    Layer L2(neurons_L1, neurons_L2);
    ActivationFcts act1, act2;
    LossFcts loss_fct;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        L1.forward(inputs);
        std::shared_ptr<Matrix> output_relu = act1.ReLU(std::make_shared<Matrix>(L1.getOutput().size(), L1.getOutput()[0].size(), L1.getOutput()));
        L2.forward(output_relu->getValues());
        std::shared_ptr<Matrix> output_softmax = act2.Softmax(std::make_shared<Matrix>(L2.getOutput().size(), L2.getOutput()[0].size(), L2.getOutput()));

        if (epoch % 1 == 0)
        {
            double loss = loss_fct.crossEntropyLoss_forward(output_softmax, y_transpose);
            std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
            losses_matrix.push_back({loss});

            std::vector<int> predictions = output_softmax->argmaxRow();
            std::vector<int> true_values = y_transpose->argmaxRow();
            int correct_predictions = 0;
            for (size_t j = 0; j < predictions.size(); ++j)
            {
                if (predictions[j] == true_values[j])
                {
                    correct_predictions++;
                }
            }
            double acc = (1.0 * correct_predictions) / (1.0 * predictions.size());
            std::cout << "Accuracy: " << acc * 100 << "%\n";
            accuracies_matrix.push_back({acc});
            predictions_matrix.push_back(predictions);
        }

        std::shared_ptr<Matrix> dinputs_loss = loss_fct.crossEntropyLoss_backward_softmax(output_softmax, y);
        L2.backward(dinputs_loss->getValues());
        std::shared_ptr<Matrix> dinputs_relu = act1.reluDerivative(std::make_shared<Matrix>(L1.getOutput().size(), L1.getOutput()[0].size(), L1.getOutput()), std::make_shared<Matrix>(L2.getDinputs().size(), L2.getDinputs()[0].size(), L2.getDinputs()));
        L1.backward(dinputs_relu->getValues());

        // Random weight and bias updates
        L1.setWeights(RandomVec(L1.getWeights().size(), L1.getWeights()[0].size()));
        L1.setBiases(RandomVec(L1.getBiases().size(), L1.getBiases()[0].size()));
        L2.setWeights(RandomVec(L2.getWeights().size(), L2.getWeights()[0].size()));
        L2.setBiases(RandomVec(L2.getBiases().size(), L2.getBiases()[0].size()));
    }

    writeToFile(losses_matrix, "losses.csv");
    writeToFile(predictions_matrix, "predictions.csv");
    writeToFile(accuracies_matrix, "accuracies.csv");
}

/**
 * @brief Trains a neural network using the Gradient Descent optimizer.
 * @param inputs Input data matrix.
 * @param neurons_L1 Number of neurons in the first layer.
 * @param neurons_L2 Number of neurons in the second layer.
 * @param epochs Number of training epochs.
 * @param learning_rate The learning rate for the optimizer.
 */
void trainNeuralNetwork_GD_Optimizer(std::vector<std::vector<double>> inputs, int neurons_L1, int neurons_L2, int epochs = 100, double learning_rate = 0.01)
{

    std::ifstream f("../data/vertical_data.json");
    if (!f.is_open())
    {
        std::cerr << "Could not open the JSON file";
        return;
    }
    json data = json::parse(f);
    f.close();

    std::vector<std::vector<double>> targets;
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
    std::vector<std::vector<int>> predictions_matrix;

    GradientDescent optimizer_GD(learning_rate);
    Layer L1(inputs[0].size(), neurons_L1);
    Layer L2(neurons_L1, neurons_L2);
    ActivationFcts act1, act2;
    LossFcts loss_fct;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        L1.forward(inputs);
        std::shared_ptr<Matrix> output_relu = act1.ReLU(std::make_shared<Matrix>(L1.getOutput().size(), L1.getOutput()[0].size(), L1.getOutput()));
        L2.forward(output_relu->getValues());
        std::shared_ptr<Matrix> output_softmax = act2.Softmax(std::make_shared<Matrix>(L2.getOutput().size(), L2.getOutput()[0].size(), L2.getOutput()));

        if (epoch % 1 == 0)
        {
            double loss = loss_fct.crossEntropyLoss_forward(output_softmax, y_transpose);
            std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
            losses_matrix.push_back({loss});

            std::vector<int> predictions = output_softmax->argmaxRow();
            std::vector<int> true_values = y_transpose->argmaxRow();
            int correct_predictions = 0;
            for (size_t j = 0; j < predictions.size(); ++j)
            {
                if (predictions[j] == true_values[j])
                {
                    correct_predictions++;
                }
            }
            double acc = (1.0*correct_predictions) / (1.0*predictions.size());
            std::cout << "Accuracy: " << acc * 100 << "%\n";
            accuracies_matrix.push_back({acc});
            predictions_matrix.push_back(predictions);
        }

        std::shared_ptr<Matrix> dinputs_loss = loss_fct.crossEntropyLoss_backward_softmax(output_softmax, y);
        L2.backward(dinputs_loss->getValues());
        std::shared_ptr<Matrix> dinputs_relu = act1.reluDerivative(std::make_shared<Matrix>(L1.getOutput().size(), L1.getOutput()[0].size(), L1.getOutput()), std::make_shared<Matrix>(L2.getDinputs().size(), L2.getDinputs()[0].size(), L2.getDinputs()));
        L1.backward(dinputs_relu->getValues());

        // Update weights and biases 
        optimizer_GD.update_params(L1);
        optimizer_GD.update_params(L2);
    }

    writeToFile(losses_matrix, "losses.csv");
    writeToFile(predictions_matrix, "predictions.csv");
    writeToFile(accuracies_matrix, "accuracies.csv");
}

int main()
{
    std::ifstream f("../data/vertical_data.json");
    if (!f.is_open())
    {
        std::cerr << "Could not open the JSON file";
        return 1;
    }
    json data = json::parse(f);
    f.close();

    std::vector<std::vector<double>> vertical_data;
    for (const auto &item : data["X"])
    {
        std::vector<double> row;
        for (const auto &value : item)
        {
            row.push_back(value.get<double>());
        }
        vertical_data.push_back(row);
    }

    trainNeuralNetwork_GD_Optimizer(vertical_data, 24, 3, 10, 0.001);
    return 0;
}
