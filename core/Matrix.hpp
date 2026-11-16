#pragma once

#include "Tensor.hpp"
#include <vector>
#include <cstddef>

class Matrix {
public:
    Matrix();
    Matrix(size_t rows, size_t cols, DeviceType device = DeviceType::CPU);
    Matrix(size_t rows, size_t cols, const std::vector<float>& data, DeviceType device = DeviceType::CPU);
    explicit Matrix(const Tensor& tensor);

    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
    size_t size() const { return m_tensor.size(); }
    DeviceType device() const { return m_tensor.device(); }

    Tensor& tensor() { return m_tensor; }
    const Tensor& tensor() const { return m_tensor; }

    float* data() { return m_tensor.data(); }
    const float* data() const { return m_tensor.data(); }

    float& operator()(size_t r, size_t c);
    const float& operator()(size_t r, size_t c) const;

    void toDevice(DeviceType dev) { m_tensor.toDevice(dev); }
    void fill(float value) { m_tensor.fill(value); }
    void zero() { m_tensor.zero(); }

    // C = alpha * op(A) * op(B) + beta * C
    static Matrix matmul(const Matrix& A, const Matrix& B, bool transA = false, bool transB = false);

private:
    size_t m_rows;
    size_t m_cols;
    Tensor m_tensor;
};

