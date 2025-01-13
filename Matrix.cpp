#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include "Matrix.h"

Matrix::Matrix(){
    numCols = 0;
    numRows = 0;
}

Matrix::Matrix(int r, int c, std::vector<std::vector<double>> v){
    numRows = r;
    numCols = c;
    values = v;
}

double getRandNum(){
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> dis(-10,10); 
    return dis(gen);
}

Matrix::Matrix(int r, int c){
    numRows = r;
    numCols = c;
    for (int i = 0; i < numRows; ++i) {
        std::vector<double> colVals;
        for (int j = 0; j < numCols; ++j) { 
            colVals.push_back(0.01 * getRandNum());
        }
        values.push_back(colVals);
    }
    //std::cout << "Rows: " << numRows << " Cols: " << numCols << std::endl;
}

Matrix::Matrix(const Matrix &m){
    *this = m;
}

    void Matrix::setValue(int r, int c, double v)
{
    values[r][c] = v;
}

void Matrix::setRowValues(int r, std::vector<double> v){
    values[r] = v;
}

int Matrix::getNumRows() const{return numRows;}
int Matrix::getNumCols() const{return numCols;}
std::vector<std::vector<double>> Matrix::getValues() const{return values;}
double Matrix::getValue(int r, int c) const{return values[r][c];}


Matrix Matrix::transpose(){
    
    Matrix m = Matrix(numCols, numRows);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            m.setValue(j ,i, getValue(i,j));
        }
        
    }

    return m;
}

Matrix Matrix::copy(){
    
    Matrix m = Matrix(numRows, numCols);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            m.setValue(i , j, getValue(i,j));
        }
        
    }

    return m;
}


void Matrix::print() const{
    //std::cout <<  " numCols" << numCols << std::endl ;
    std::cout <<  "[ " ;
    for (int i = 0; i < numRows; i++) {
    //for (int i = 0; i < 5; i++) {
        std::cout <<  "[ " ;
        for (int j = 0; j < numCols; j++) { 
            std::cout << values[i][j] << " " ; 
        }
        std::cout <<  " ]" << std::endl ;
    }
    
    std::cout <<  " ]" << std::endl ;
    std::cout <<  " end" << std::endl ;
}

Matrix Matrix::operator-(const Matrix& mat2) const{
    if(numRows != mat2.numRows || numCols != mat2.numCols) {
        throw std::invalid_argument("The dimensions do not match");
    }

    Matrix result = Matrix(numRows, numCols);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            result.setValue(i ,j, getValue(i,j) - mat2.getValue(i,j));
        }
        
    }

    return result;


}

Matrix Matrix::operator+(const Matrix& mat2) const{
    if(numRows != mat2.numRows || numCols != mat2.numCols) {
        throw std::invalid_argument("The dimensions do not match");
    }

    Matrix result = Matrix(numRows, numCols);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            result.setValue(i ,j, getValue(i,j) + mat2.getValue(i,j));
        }
        
    }
    return result;
}

Matrix Matrix::operator+=(const Matrix &mat2){
    *this = *this + mat2;
    return *this;
}

Matrix Matrix::operator*(const Matrix &mat2) const
{
    if(numRows != mat2.numRows || numCols != mat2.numCols) {
        throw std::invalid_argument("The dimensions do not match");
    }

    Matrix result = Matrix(numRows, numCols);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            result.setValue(i ,j, getValue(i,j) * mat2.getValue(i,j));
        }
        
    }

    return result;
}

Matrix Matrix::operator*(const double &k) const
{
    Matrix result = Matrix(numRows, numCols);
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {

            result.setValue(i, j, getValue(i, j) * k);
        }
    }

    return result;
}

    int Matrix::operator==(const Matrix &mat2) const
{
    if(numCols!=mat2.getNumCols() || numRows!=mat2.getNumRows() || values!=mat2.values){
        return 0;
    }
    return 1;
}

Matrix& Matrix::operator=(const Matrix &mat2){
    numCols = mat2.getNumCols();
    numRows = mat2.getNumRows();
    values = mat2.getValues();
    return *this;
}

Matrix& Matrix::operator=(const std::vector<std::vector<double>> mat2){
    numCols = mat2[0].size();
    numRows = mat2.size();
    values = mat2;
    return *this;
}

    Matrix Matrix::operator/(const Matrix &mat2) const
{
    if(numRows != mat2.numRows || numCols != mat2.numCols) {
        throw std::invalid_argument("The dimensions do not match");
    }

    Matrix result = Matrix(numRows, numCols);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            result.setValue(i ,j, getValue(i,j) / mat2.getValue(i,j));
        }
        
    }

    return result;
}

Matrix Matrix::divide(int den) const{
    
    Matrix result = Matrix(numRows, numCols);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            result.setValue(i ,j, getValue(i,j) / den);
        }
        
    }

    return result;
}

/**
 * @brief Performs matrix multiplication (dot product) between the current matrix and another matrix.
 *
 * This function calculates the dot product of the current matrix with the matrix provided as an argument.
 * The matrix multiplication requires that the number of columns in the current matrix must be equal to the
 * number of rows in the provided matrix. If this condition is not met, the function throws an exception.
 * The result is a new matrix with dimensions equal to the number of rows of the current matrix and the number of columns
 * of the provided matrix.
 *
 * The dot product of two matrices A (m x n) and B (n x p) results in a matrix C (m x p), where each element
 * C(i,j) is computed as the sum of the products of corresponding elements from row i of matrix A and column j of matrix B.
 *
 * @param mat The matrix to multiply with the current matrix. It must have the same number of rows as the number of columns
 *            of the current matrix.
 * @return A new matrix resulting from the dot product of the current matrix and the provided matrix.
 *
 */
Matrix Matrix::dotProduct(const Matrix& mat) const{

    if (numCols != mat.numRows)
    {
        std::cout << "this.cols() : " << numCols << " mat->getNumRows() : " << mat.getNumRows() << std::endl;
        throw std::invalid_argument("Invalid dimensions. Remember to use the transpose of the mat_weights matrix");
    }

    int r = numRows;
    int c = mat.numCols;

    Matrix result = Matrix(numRows,mat.numCols);

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            double sum = 0;
            for (int k = 0; k < numCols; ++k) {
                sum += getValue(i,k) * mat.getValue(k,j);
            }
            result.setValue(i,j, sum);
        }
    }

    return result;
}

std::vector<double> Matrix::sumMat() const{

    std::vector<double> result;
    
    for (int i = 0; i < numRows; ++i) {
        double sum = 0;
        for (int j = 0; j < numCols; ++j) { 
            sum += values[i][j];
            
        }
        result.push_back(sum);
    }

    return result;

}

std::vector<std::vector<double>> Matrix::sumRowsMat() const{
    std::vector<std::vector<double>> result(numRows, std::vector<double>(numCols, 0.0));

    
    for (int i = 0; i < numRows; ++i) {
        double sum = 0;
        for (int j = 0; j < numCols; ++j) { 
            sum += values[i][j];
        }
        std::fill(result[i].begin(), result[i].end(), sum);
    }

    return result;
}

int Matrix::argmax(const std::vector<double>& vec) const{
    return std::distance(vec.begin(), std::max_element(vec.begin(),vec.end()));
}

std::vector<int> Matrix::argmaxRow() const{
    std::vector<int> result;
    for (auto &row : values) {
        result.push_back(argmax(row));
    }

    return result;
}

//axis = 0
std::vector<std::vector<double>> Matrix::sumOverRows() const{
    std::vector<double> result(numCols, 0.0);

    
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
            result[j] += values[i][j];
        }
    }

    return {result};
}

//axis = 1
std::vector<std::vector<double>> Matrix::sumOverCols() const{
    std::vector<std::vector<double>> result(numRows, std::vector<double>(1, 0.0));

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
            result[0][i] += values[i][j];
        }
    }

    return result;
}
