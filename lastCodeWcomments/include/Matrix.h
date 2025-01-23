#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <memory>

/**
 * @class Matrix
 * @brief Represents a matrix with various operations such as dot product, transpose, and element-wise arithmetic.
 */
class Matrix
{
private:
    int numRows;                             ///< Number of rows in the matrix.
    int numCols;                             ///< Number of columns in the matrix.
    std::vector<std::vector<double>> values; ///< 2D vector to store matrix values.

public:
    /**
     * @brief Constructs a matrix with given dimensions and values.
     * @param r Number of rows.
     * @param c Number of columns.
     * @param v Values to initialize the matrix.
     */
    Matrix(int r, int c, std::vector<std::vector<double>> v);

    /**
     * @brief Constructs a matrix with given dimensions and random values initialized using a normal distribution.
     * @param r Number of rows.
     * @param c Number of columns.
     */
    Matrix(int r, int c);

    /**
     * @brief Sets the value at a specific position in the matrix.
     * @param r Row index.
     * @param c Column index.
     * @param v Value to set.
     */
    void setValue(int r, int c, double v);

    /**
     * @brief Sets the values for a specific row in the matrix.
     * @param r Row index.
     * @param v Vector of values to set for the row.
     */
    void setRowValues(int r, std::vector<double> v);

    /**
     * @brief Gets the number of rows in the matrix.
     * @return The number of rows.
     */
    int getNumRows() const;

    /**
     * @brief Gets the number of columns in the matrix.
     * @return The number of columns.
     */
    int getNumCols() const;

    /**
     * @brief Gets all values of the matrix.
     * @return 2D vector containing all matrix values.
     */
    std::vector<std::vector<double>> getValues() const;

    /**
     * @brief Gets the value at a specific position in the matrix.
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
     * @brief Creates a copy of the current matrix.
     * @return A shared pointer to the new matrix.
     */
    std::shared_ptr<Matrix> copy();

    /**
     * @brief Subtracts the current matrix to another matrix element-wise.
     * @param mat2 The matrix to subtract.
     * @return A shared pointer to the resulting matrix.
     */
    std::shared_ptr<Matrix> operator-(const Matrix &mat2) const;

    /**
     * @brief Divides the current matrix element-wise by another matrix.
     * @param mat2 The matrix to divide by.
     * @return A shared pointer to the resulting matrix.
     */
    std::shared_ptr<Matrix> operator/(const Matrix &mat2) const;

    /**
     * @brief Divides all elements of the matrix by a scalar.
     * @param den Scalar value to divide by.
     * @return A shared pointer to the resulting matrix.
     */
    std::shared_ptr<Matrix> divide(int den) const;

    /**
     * @brief Adds the current matrix to another matrix element-wise.
     * @param mat2 The matrix to add.
     * @return A shared pointer to the resulting matrix.
     */
    std::shared_ptr<Matrix> operator+(const Matrix &mat2) const;

    /**
     * @brief Multiplies the current matrix to another matrix element-wise.
     * @param mat2 The matrix to multiply by.
     * @return A shared pointer to the resulting matrix.
     */
    std::shared_ptr<Matrix> operator*(const Matrix &mat2) const;

    /**
     * @brief Performs the dot product with another matrix.
     * @param mat The matrix to perform the dot product with.
     * @return A shared pointer to the resulting matrix.
     * @throws std::invalid_argument if dimensions are incompatible.
     */
    std::shared_ptr<Matrix> dotProduct(const Matrix &mat) const;

    /**
     * @brief Sums all elements in each row of the matrix.
     * @return A vector containing the sum of each row.
     */
    std::vector<double> sumMat() const;

    /**
     * @brief Computes the sum of elements along each row and returns a matrix.
     * @return A matrix containing the row-wise sums replicated for all columns.
     */
    std::vector<std::vector<double>> sumRowsMat() const;

    /**
     * @brief Finds the index of the maximum value in a vector.
     * @param vec The input vector.
     * @return The index of the maximum value.
     */
    int argmax(const std::vector<double> &vec) const;

    /**
     * @brief Finds the index of the maximum value for each row.
     * @return A vector containing the index of the maximum value for each row.
     */
    std::vector<double> argmaxRow() const;

    /**
     * @brief Sums all elements column-wise and returns a row vector.
     * @return A matrix with one row containing the column-wise sums.
     */
    std::vector<std::vector<double>> sumOverRows() const;

    /**
     * @brief Sums all elements row-wise and returns a column vector.
     * @return A matrix with one column containing the row-wise sums.
     */
    std::vector<std::vector<double>> sumOverCols() const;
};

#endif // MATRIX_H
