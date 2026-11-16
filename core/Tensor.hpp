#pragma once

#include "CUDABackend.hpp"
#include <vector>
#include <cstddef>
#include <string>
#include <initializer_list>

class Tensor {
public:
    Tensor();
    explicit Tensor(const std::vector<size_t>& shape, DeviceType device = DeviceType::CPU);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& initialData, DeviceType device = DeviceType::CPU);
    Tensor(const std::vector<size_t>& shape, float fillValue, DeviceType device = DeviceType::CPU);

    ~Tensor();

    // Copy semantics (deep copy)
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    // Move semantics
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Attributes
    const std::vector<size_t>& shape() const { return m_shape; }
    const std::vector<size_t>& strides() const { return m_strides; }
    size_t size() const { return m_size; }
    size_t dim() const { return m_shape.size(); }
    DeviceType device() const { return m_device; }
    bool empty() const { return m_size == 0 || m_data == nullptr; }

    // Memory pointers
    float* data() { return m_data; }
    const float* data() const { return m_data; }

    // Flat indexing (Host / CPU only)
    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    // Multi-dimensional indexing (Host / CPU only)
    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;

    // Device transfers
    void toDevice(DeviceType targetDevice);
    Tensor to(DeviceType targetDevice) const;

    // In-place initialization & modification
    void fill(float value);
    void zero();

    // Data transfer helpers
    void copyFrom(const Tensor& other);
    void copyFromHost(const float* hostData, size_t count);
    void copyToHost(float* hostData, size_t count) const;
    std::vector<float> toHostVector() const;

    // Reshaping (must preserve total number of elements)
    Tensor reshape(const std::vector<size_t>& newShape) const;

    // Utility
    void print(const std::string& name = "", size_t maxElements = 16) const;

private:
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;
    size_t m_size;
    DeviceType m_device;
    float* m_data;

    void computeStrides();
    void allocate();
    void deallocate();
};

