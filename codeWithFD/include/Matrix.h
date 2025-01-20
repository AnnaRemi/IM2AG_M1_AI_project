#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <memory>


class Matrix{
    private:
        int numRows;
        int numCols;
        std::vector<std::vector<double>> values;

    public:
        Matrix(int r, int c, std::vector<std::vector<double>> v );
        Matrix(int r, int c);

        void setValue(int r, int c, double v);
        void setRowValues(int r, std::vector<double> v);
        int getNumRows() const;
        int getNumCols() const;
        std::vector<std::vector<double>> getValues() const;
        double getValue(int r, int c) const;
        void print();

        std::shared_ptr<Matrix> transpose();
        std::shared_ptr<Matrix> copy();
        std::shared_ptr<Matrix> operator-(const Matrix& mat2) const;
        std::shared_ptr<Matrix> operator/(const Matrix& mat2) const;
        std::shared_ptr<Matrix> divide(int den) const;
        std::shared_ptr<Matrix> operator+(const Matrix& mat2) const;
        std::shared_ptr<Matrix> operator+(double) const;
        std::shared_ptr<Matrix> operator*(const Matrix& mat2) const;
        //std::shared_ptr<Matrix> dotProduct(const Matrix& mat) const;
        std::shared_ptr<Matrix> dotProduct(const Matrix& mat) const;
        std::vector<double> sumMat() const;
        std::vector<std::vector<double>> sumRowsMat() const;
        int argmax(const std::vector<double>& vec) const;
        std::vector<double> argmaxRow() const;
        std::vector<std::vector<double>> sumOverRows() const;
        std::vector<std::vector<double>> sumOverCols() const;
        

    

};

#endif // MATRIX_H