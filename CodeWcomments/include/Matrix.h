#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <memory>

/**
 * @class Matrix
 * @brief Represents a matrix with basic operations and manipulation methods.
 */
class Matrix
{
private:
    int numRows;                             ///< Number of rows in the matrix.
    int numCols;                             ///< Number of columns in the matrix.
    std::vector<std::vector<double>> values; ///< Values stored in the matrix.

public:
    /**
     * @brief Constructs a matrix with specified dimensions and values.
     * @param r Number of rows.
     * @param c Number of columns.
     * @param v Values to initialize the matrix.
     */
    Matrix(int r, int c, std::vector<std::vector<double>> v);

    /**
     * @brief Constructs a matrix with specified dimensions and random values.
     * @param r Number of rows.
     * @param c Number of columns.
     */
    Matrix(int r, int c);

    /**
     * @brief Sets the value of a specific element.
     * @param r Row index.
     * @param c Column index.
     * @param v Value to set.
     */
    void setValue(int r, int c, double v);

    /**
     * @brief Sets the values for a specific row.
     * @param r Row index.
     * @param v Vector of values for the row.
     */
    void setRowValues(int r, std::vector<double> v);

    /**
     * @brief Returns the number of rows.
     * @return Number of rows in the matrix.
     */
    int getNumRows() const;

    /**
     * @brief Returns the number of columns.
     * @return Number of columns in the matrix.
     */
    int getNumCols() const;

    /**
     * @brief Returns the values of the matrix.
     * @return 2D vector containing the matrix values.
     */
    std::vector<std::vector<double>> getValues() const;

    /**
     * @brief Returns the value at a specific position.
     * @param r Row index.
     * @param c Column index.
     * @return Value at the specified position.
     */
    double getValue(int r, int c) const;

    /**
     * @brief Prints the matrix values.
     */
    void print();

    /**
     * @brief Returns the transpose of the matrix.
     * @return A shared pointer to the transposed matrix.
     */
    std::shared_ptr<Matrix> transpose();

    /**
     * @brief Creates a copy of the matrix.
     * @return A shared pointer to the copied matrix.
     */
    std::shared_ptr<Matrix> copy();

    /**
     * @brief Overloaded subtraction operator for matrix subtraction.
     * @param mat2 The matrix to subtract.
     * @return A shared pointer to the resulting matrix.
     */
    std::shared_ptr<Matrix> operator-(const Matrix &mat2) const;

    /**
     * @brief Overloaded division operator for element-wise division.
     * @param mat2 The matrix to divide by.
     * @return A shared pointer to the resulting matrix.
     */
    std::shared_ptr<Matrix> operator/(const Matrix &mat2) const;

    /**
     * @brief Divides all elements of the matrix by a scalar.
     * @param den The scalar value to divide by.
     * @return A shared pointer to the resulting matrix.
     */
    std::shared_ptr<Matrix> divide(int den) const;

    /**
     * @brief Overloaded addition operator for matrix addition.
     * @param mat2 The matrix to add.
     * @return A shared pointer to the resulting matrix.
     */
    std::shared_ptr<Matrix> operator+(const Matrix &mat2) const;

    /**
     * @brief Overloaded multiplication operator for element-wise multiplication.
     * @param mat2 The matrix to multiply by.
     * @return A shared pointer to the resulting matrix.
     */
    std::shared_ptr<Matrix> operator*(const Matrix &mat2) const;

    /**
     * @brief Computes the dot product of the current matrix with another matrix.
     * @param mat The matrix to perform the dot product with.
     * @return A shared pointer to the resulting matrix.
     */
    std::shared_ptr<Matrix> dotProduct(const Matrix &mat) const;

    /**
     * @brief Computes the sum of elements in each row.
     * @return A vector containing the sums of each row.
     */
    std::vector<double> sumMat() const;

    /**
     * @brief Computes the sum of elements along each row and returns a matrix.
     * @return A matrix containing the sums of each row.
     */
    std::vector<std::vector<double>> sumRowsMat() const;

    /**
     * @brief Returns the index of the maximum value in a vector.
     * @param vec The input vector.
     * @return The index of the maximum value.
     */
    int argmax(const std::vector<double> &vec) const;

    /**
     * @brief Returns the index of the maximum value for each row.
     * @return A vector containing the index of the maximum value for each row.
     */
    std::vector<int> argmaxRow() const;

    /**
     * @brief Computes the sum of elements along the rows.
     * @return A matrix with the sum over rows.
     */
    std::vector<std::vector<double>> sumOverRows() const;

    /**
     * @brief Computes the sum of elements along the columns.
     * @return A matrix with the sum over columns.
     */
    std::vector<std::vector<double>> sumOverCols() const;
};

#endif // MATRIX_H