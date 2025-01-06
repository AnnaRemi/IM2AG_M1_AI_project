#include "Layer_version3.hpp"

Layer::Layer(int n_input, int n_neurons)
{
    neurons = MatrixXd::Random(n_input, n_neurons);
    biases = VectorXd::Zero(n_neurons);
}

Layer::Layer(MatrixXd n, VectorXd b)
{
    if (b.size() != n.cols())
    {
        throw std::invalid_argument("Invalid dimensions : The number of elements of biases has to be equal to the number of columns of neurons");
    }
    neurons = n;
    biases = b;
}

MatrixXd Layer::forward(MatrixXd inputs)
{
    if (inputs.cols() != neurons.rows())
    {
        throw std::invalid_argument("Invalid dimensions : The number of columns of the input matrix has to be equal to the number of lines of the neurons matrix");
    }
    MatrixXd mat_biases(inputs.rows(), biases.size());
    for (int i = 0; i < inputs.rows(); i++)
    {
        mat_biases.row(i) = biases.transpose();
    }
    return (inputs * neurons) + mat_biases;
}

std::ostream &operator<<(std::ostream &o, const Layer &L)
{
    o << "Weights : " << std::endl;
    o << L.neurons << std::endl;
    o << "Biases : " << std::endl;
    o << L.biases << std::endl;
    return o;
}

MatrixXd activation_ReLU(MatrixXd input){
    return input.cwiseMax(0);
}

MatrixXd activation_softmax(MatrixXd input){
    MatrixXd output = input.array().exp();
    VectorXd sum_rows = output.rowwise().sum();
    for (int i = 0; i < input.rows(); i++){
        output.row(i) /= sum_rows(i);
    }
    return output;
}

MatrixXd loss(MatrixXd y_pred, MatrixXd y_true){
    MatrixXd correct_confidence_mat;
    VectorXd correct_confidence_vec;
    int samples = y_pred.rows();
    /*y_pred_clipped is the matrix y_pred with all coeffs being between 1e-7 and 1 - 1e-7 (if the coeff is 0 it will be 
    replaced by 1e-7 and if it is 1 it will be replaced by 1 - 1e-7)    
    */
    MatrixXd y_pred_clipped = y_pred.cwiseMax(1e-7);
    y_pred_clipped = y_pred_clipped.cwiseMin(1 - 1e-7);
    // y_true.rows() == 1 <=> y_true is a vector <=> len(y_true.shape == 1)
    if(y_true.rows() == 1){

    }
    //if y_true.rows() != 1 then y_true.shape ==2
    else{
        correct_confidence_mat = y_pred.cwiseProduct(y_true);
        correct_confidence_vec = correct_confidence_mat.rowwise().sum().transpose();//the sum rowwise will give us a a column matrix so we take the transpose to have a vector
    }
    return (-correct_confidence_vec.array()).log().matrix();
}
