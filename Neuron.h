#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <iostream>
#include <stdexcept>

/*template <typename T>
using objectType = std::conditional_t<std::is_same_v<T, double>, std::vector<double>, std::vector<std::vector<double>>>;*/

/*template <typename T>
class Neuron{
    private:
        T weights;
        T inputs;
        double bias;

    public:
        Neuron();
        Neuron(T w, T i, double b);

        void setWeights(T w);

        double getBias();
        double dotProduct();
        double output();  

};*/


class Neuron{
    private:
        std::vector<std::vector<double>> weights;
        std::vector<std::vector<double>> inputs;
        double bias;

    public:
        Neuron();
        Neuron(std::vector<std::vector<double>> w, std::vector<std::vector<double>> i, double b);

        void setWeights(std::vector<std::vector<double>> w);

        double getBias();
        std::vector<std::vector<double>> dotProduct();
        std::vector<std::vector<double>> output();  

};

#endif // NEURON_H