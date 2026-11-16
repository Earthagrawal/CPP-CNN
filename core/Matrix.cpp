#include "Matrix.hpp"
#include <stdexcept>

Matrix::Matrix() : m_rows(0), m_cols(0), m_tensor() {}

Matrix::Matrix(size_t rows, size_t cols, DeviceType device)
    : m_rows(rows), m_cols(cols), m_tensor({rows, cols}, device) {}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<float>& data, DeviceType device)
    : m_rows(rows), m_cols(cols), m_tensor({rows, cols}, data, device) {}

Matrix::Matrix(const Tensor& tensor) : m_tensor(tensor) {
    if (tensor.shape().size() != 2) {
        throw std::invalid_argument("Cannot construct Matrix from non-2D Tensor");
    }
    m_rows = tensor.shape()[0];
    m_cols = tensor.shape()[1];
}

float& Matrix::operator()(size_t r, size_t c) {
    return m_tensor.at({r, c});
}

const float& Matrix::operator()(size_t r, size_t c) const {
    return m_tensor.at({r, c});
}

Matrix Matrix::matmul(const Matrix& A, const Matrix& B, bool transA, bool transB) {
    size_t M = transA ? A.cols() : A.rows();
    size_t K_A = transA ? A.rows() : A.cols();
    size_t K_B = transB ? B.cols() : B.rows();
    size_t N = transB ? B.rows() : B.cols();

    if (K_A != K_B) {
        throw std::invalid_argument("Inner matrix dimensions do not match for multiplication");
    }
    if (A.device() != B.device()) {
        throw std::invalid_argument("Matrix devices do not match for multiplication");
    }

    Matrix C(M, N, A.device());

    int lda = static_cast<int>(A.cols());
    int ldb = static_cast<int>(B.cols());
    int ldc = static_cast<int>(N);

    CUDABackend::sgemm(transA, transB,
                       static_cast<int>(M), static_cast<int>(N), static_cast<int>(K_A),
                       1.0f,
                       A.data(), lda,
                       B.data(), ldb,
                       0.0f,
                       C.data(), ldc,
                       A.device());

    return C;
}

