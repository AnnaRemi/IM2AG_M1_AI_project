#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include "../include/Matrix.h"

Matrix::Matrix(int r, int c, std::vector<std::vector<double>> v)
{
    if (v.size() > 0 && c != v[0].size())
    {
        throw std::invalid_argument("Number of columns does not match the provided size.");
    }
    if (r != v.size())
    {
        throw std::invalid_argument("Number of rows does not match the provided size.");
    }

    numRows = r;
    numCols = c;
    values = std::move(v); // Use move semantics for better performance
}

double getRandNum(){
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> dis(-100,100); 
    return dis(gen);
}

Matrix::Matrix(int r, int c){
    numRows = r;
    numCols = c;
    values.resize( numRows );
    for (int i = 0; i < numRows; ++i) {
        values[i].resize( numCols );
        for (int j = 0; j < numCols; ++j) { 
            values[i][j] = 0.01 * getRandNum();
        }
    }
}


void Matrix::setValue(int r, int c, double v){
    values[r][c] = v;
}

void Matrix::setRowValues(int r, std::vector<double> v){
    values[r] = v;
}

int Matrix::getNumRows() const{return numRows;}
int Matrix::getNumCols() const{return numCols;}
std::vector<std::vector<double>> Matrix::getValues() const{return values;}
double Matrix::getValue(int r, int c) const{return values[r][c];}

std::shared_ptr<Matrix> Matrix::transpose(){
    std::shared_ptr<Matrix> m = std::make_shared<Matrix>(numCols, numRows);
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            m->setValue(j ,i, getValue(i,j));
        }
        
    }

    return m;
}

std::shared_ptr<Matrix> Matrix::copy(){
    
    std::shared_ptr<Matrix> m = std::make_shared<Matrix>(numRows, numCols);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            m->setValue(i , j, getValue(i,j));
        }
        
    }

    return m;
}


void Matrix::print(){
   
    std::cout <<  "[ " ;
    for (int i = 0; i < numRows; i++) {
        std::cout <<  "[ " ;
        for (int j = 0; j < numCols; j++) { 
            std::cout << values[i][j] << " " ; 
        }
        std::cout <<  " ]" << std::endl ;
    }
    
    std::cout <<  " ]" << std::endl ;
    std::cout <<  " end" << std::endl ;
}

std::shared_ptr<Matrix> Matrix::operator-(const Matrix& mat2) const{
    if(numRows != mat2.numRows || numCols != mat2.numCols) {
        throw std::invalid_argument("The dimensions do not match");
    }


    std::shared_ptr<Matrix> result = std::make_shared<Matrix>(numRows, numCols);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            result->setValue(i ,j, getValue(i,j) - mat2.getValue(i,j));
        }
        
    }

    return result;


}

std::shared_ptr<Matrix> Matrix::operator+(const Matrix& mat2) const{
    if(numRows != mat2.numRows || numCols != mat2.numCols) {
        throw std::invalid_argument("The dimensions do not match");
    }

    std::shared_ptr<Matrix> result = std::make_shared<Matrix>(numRows, numCols);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            result->setValue(i ,j, getValue(i,j) + mat2.getValue(i,j));
        }
        
    }

    return result;


}

std::shared_ptr<Matrix> Matrix::operator+(double d) const{
    std::shared_ptr<Matrix> result = std::make_shared<Matrix>(numRows, numCols);
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            result->setValue(i ,j, getValue(i,j) + d);
        }
        
    }

    return result;
}



std::shared_ptr<Matrix> Matrix::operator*(const Matrix& mat2) const{
    if(numRows != mat2.numRows || numCols != mat2.numCols) {
        throw std::invalid_argument("The dimensions do not match");
    }

    std::shared_ptr<Matrix> result = std::make_shared<Matrix>(numRows, numCols);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            result->setValue(i ,j, getValue(i,j) * mat2.getValue(i,j));
        }
        
    }

    return result;

}

std::shared_ptr<Matrix> Matrix::operator/(const Matrix& mat2) const{
    if(numRows != mat2.numRows || numCols != mat2.numCols) {
        throw std::invalid_argument("The dimensions do not match");
    }

    std::shared_ptr<Matrix> result = std::make_shared<Matrix>(numRows, numCols);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            result->setValue(i ,j, getValue(i,j) / mat2.getValue(i,j));
        }
        
    }

    return result;
}

std::shared_ptr<Matrix> Matrix::divide(int den) const{
    
    std::shared_ptr<Matrix> result = std::make_shared<Matrix>(numRows, numCols);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) { 
     
            result->setValue(i ,j, getValue(i,j) / den);
        }
        
    }

    return result;
}


std::shared_ptr<Matrix> Matrix::dotProduct(const Matrix& mat) const{

    if (numCols != mat.numRows)
    {
        std::cout << "this.cols() : " << numCols << " mat->getNumRows() : " << mat.getNumRows() << std::endl;
        throw std::invalid_argument("Invalid dimensions. Remember to use the transpose of the mat_weights matrix");
    }

    int r = numRows;
    int c = mat.numCols;

    std::shared_ptr<Matrix> result = std::make_shared<Matrix>(numRows,mat.numCols);

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            double sum = 0;
            for (int k = 0; k < numCols; ++k) {
                sum += getValue(i,k) * mat.getValue(k,j);
            }
            result->setValue(i,j, sum);
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

std::vector<double> Matrix::argmaxRow() const{
    std::vector<double> result;
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


