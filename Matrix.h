#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>


class Matrix{
    private:
        int numRows;
        int numCols;
        std::vector<std::vector<double>> values;

    public:
        Matrix();
        Matrix(int r, int c, std::vector<std::vector<double>> v);
        Matrix(int r, int c);
        Matrix(const Matrix &m);

        void setValue(int r, int c, double v);
        void setRowValues(int r, std::vector<double> v);
        int getNumRows() const;
        int getNumCols() const;
        std::vector<std::vector<double>> getValues() const;
        double getValue(int r, int c) const;
        void print() const;
        Matrix transpose();
        Matrix copy();
        Matrix operator-(const Matrix& mat2) const;
        Matrix operator/(const Matrix& mat2) const;
        Matrix divide(int den) const;
        Matrix operator+(const Matrix& mat2) const;
        Matrix operator+=(const Matrix &mat2);
        Matrix operator*(const Matrix &mat2) const;
        Matrix operator*(const double &k) const;
        int operator==(const Matrix &mat2) const;
        Matrix& operator=(const Matrix &mat2);
        Matrix &operator=(const std::vector<std::vector<double>> mat2);
        Matrix dotProduct(const Matrix &mat) const;
        std::vector<double> sumMat() const;
        std::vector<std::vector<double>> sumRowsMat() const;
        int argmax(const std::vector<double>& vec) const;
        std::vector<int> argmaxRow() const;
        std::vector<std::vector<double>> sumOverRows() const;
        std::vector<std::vector<double>> sumOverCols() const;
        
};

#endif // MATRIX_H
