#include "Layer_version3.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void ex_forward_pass(int n_neurons_first_layer, int n_neurons_second_layer, int n_inputs){
    //Construct random layers and random input
    Layer L1(n_inputs, n_neurons_first_layer);
    MatrixXd inputs = MatrixXd::Random(1, n_inputs);
    Layer L2(n_neurons_first_layer, n_neurons_second_layer);
    MatrixXd output1, output2;
    //First Layer
    output1 = activation_ReLU(L1.forward(inputs));
    //Second Layer
    output2 = activation_softmax(L2.forward(output1));
    //Print
    std::cout << "First Layer : " << L1 << std::endl;
    std::cout << "Second Layer : " << L2 << std::endl;
    std::cout << "Input : " << inputs << std::endl;
    std::cout << "Final output : " << output2 << std::endl;
}

int main(){
    // Layer L1(2, 3);
    // MatrixXd neurons{{1, 2, 3}, {4, 5, 6}};
    // VectorXd biases(3);
    // biases << 1, 1, 1;
    // Layer L2(neurons, biases);
    // MatrixXd Inputs(4, 2);
    // Inputs << 1, 2,
    //     -5, 1,
    //     1, 1,
    //     1, -1;
    // std::cout << activation_ReLU(Inputs) << std::endl;
    // std::cout << L2;
    // std::cout << L2.forward(Inputs) << std::endl;
    //ex_forward_pass(5, 4, 6);
    MatrixXd l = MatrixXd::Random(5, 5);
    MatrixXd c;
    c = l.rowwise().sum().transpose();
    std::cout << l << std::endl;
    std::cout << c << std::endl;
    return 0;
}