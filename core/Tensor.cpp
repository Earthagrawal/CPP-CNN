#include "Tensor.hpp"
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <algorithm>

Tensor::Tensor()
    : m_shape({}), m_strides({}), m_size(0), m_device(DeviceType::CPU), m_data(nullptr) {}

Tensor::Tensor(const std::vector<size_t>& shape, DeviceType device)
    : m_shape(shape), m_device(device), m_data(nullptr) {
    computeStrides();
    allocate();
    zero();
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& initialData, DeviceType device)
    : m_shape(shape), m_device(device), m_data(nullptr) {
    computeStrides();
    allocate();
    if (initialData.size() != m_size) {
        throw std::invalid_argument("Initial data size does not match Tensor shape size");
    }
    copyFromHost(initialData.data(), m_size);
}

Tensor::Tensor(const std::vector<size_t>& shape, float fillValue, DeviceType device)
    : m_shape(shape), m_device(device), m_data(nullptr) {
    computeStrides();
    allocate();
    fill(fillValue);
}

Tensor::~Tensor() {
    deallocate();
}

void Tensor::computeStrides() {
    m_size = 1;
    if (m_shape.empty()) {
        m_size = 0;
        m_strides.clear();
        return;
    }
    m_strides.resize(m_shape.size());
    for (size_t s : m_shape) {
        m_size *= s;
    }
    size_t stride = 1;
    for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i) {
        m_strides[i] = stride;
        stride *= m_shape[i];
    }
}

void Tensor::allocate() {
    if (m_size == 0) {
        m_data = nullptr;
        return;
    }
    if (m_device == DeviceType::CPU) {
        m_data = new float[m_size]();
    } else {
        m_data = CUDABackend::allocateDevice(m_size);
    }
}

void Tensor::deallocate() {
    if (m_data != nullptr) {
        if (m_device == DeviceType::CPU) {
            delete[] m_data;
        } else {
            CUDABackend::freeDevice(m_data);
        }
        m_data = nullptr;
    }
}

Tensor::Tensor(const Tensor& other)
    : m_shape(other.m_shape), m_strides(other.m_strides), m_size(other.m_size), m_device(other.m_device), m_data(nullptr) {
    if (m_size > 0 && other.m_data != nullptr) {
        allocate();
        copyFrom(other);
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate();
        m_shape = other.m_shape;
        m_strides = other.m_strides;
        m_size = other.m_size;
        m_device = other.m_device;
        if (m_size > 0 && other.m_data != nullptr) {
            allocate();
            copyFrom(other);
        }
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : m_shape(std::move(other.m_shape)),
      m_strides(std::move(other.m_strides)),
      m_size(other.m_size),
      m_device(other.m_device),
      m_data(other.m_data) {
    other.m_size = 0;
    other.m_data = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        m_shape = std::move(other.m_shape);
        m_strides = std::move(other.m_strides);
        m_size = other.m_size;
        m_device = other.m_device;
        m_data = other.m_data;

        other.m_size = 0;
        other.m_data = nullptr;
    }
    return *this;
}

float& Tensor::operator[](size_t index) {
    if (m_device != DeviceType::CPU) {
        throw std::runtime_error("Direct indexing [] is only supported on CPU tensors.");
    }
    if (index >= m_size) {
        throw std::out_of_range("Index out of range");
    }
    return m_data[index];
}

const float& Tensor::operator[](size_t index) const {
    if (m_device != DeviceType::CPU) {
        throw std::runtime_error("Direct indexing [] is only supported on CPU tensors.");
    }
    if (index >= m_size) {
        throw std::out_of_range("Index out of range");
    }
    return m_data[index];
}

float& Tensor::at(const std::vector<size_t>& indices) {
    if (m_device != DeviceType::CPU) {
        throw std::runtime_error("Tensor::at() is only supported on CPU tensors.");
    }
    if (indices.size() != m_shape.size()) {
        throw std::invalid_argument("Incorrect number of indices");
    }
    size_t flatIndex = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= m_shape[i]) {
            throw std::out_of_range("Index out of range in dimension " + std::to_string(i));
        }
        flatIndex += indices[i] * m_strides[i];
    }
    return m_data[flatIndex];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    if (m_device != DeviceType::CPU) {
        throw std::runtime_error("Tensor::at() is only supported on CPU tensors.");
    }
    if (indices.size() != m_shape.size()) {
        throw std::invalid_argument("Incorrect number of indices");
    }
    size_t flatIndex = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= m_shape[i]) {
            throw std::out_of_range("Index out of range in dimension " + std::to_string(i));
        }
        flatIndex += indices[i] * m_strides[i];
    }
    return m_data[flatIndex];
}

void Tensor::toDevice(DeviceType targetDevice) {
    if (m_device == targetDevice || m_size == 0) return;

    if (m_device == DeviceType::CPU && targetDevice == DeviceType::CUDA) {
        float* devPtr = CUDABackend::allocateDevice(m_size);
        CUDABackend::copyHostToDevice(devPtr, m_data, m_size);
        delete[] m_data;
        m_data = devPtr;
        m_device = DeviceType::CUDA;
    } else if (m_device == DeviceType::CUDA && targetDevice == DeviceType::CPU) {
        float* hostPtr = new float[m_size]();
        CUDABackend::copyDeviceToHost(hostPtr, m_data, m_size);
        CUDABackend::freeDevice(m_data);
        m_data = hostPtr;
        m_device = DeviceType::CPU;
    }
}

Tensor Tensor::to(DeviceType targetDevice) const {
    Tensor copy(*this);
    copy.toDevice(targetDevice);
    return copy;
}

void Tensor::fill(float value) {
    if (m_size == 0 || m_data == nullptr) return;
    if (m_device == DeviceType::CPU) {
        std::fill(m_data, m_data + m_size, value);
    } else {
        CUDABackend::setDeviceMemory(m_data, value, m_size);
    }
}

void Tensor::zero() {
    fill(0.0f);
}

void Tensor::copyFrom(const Tensor& other) {
    if (m_size != other.m_size) {
        throw std::invalid_argument("Tensor sizes must match for copyFrom");
    }
    if (m_device == DeviceType::CPU && other.m_device == DeviceType::CPU) {
        std::memcpy(m_data, other.m_data, m_size * sizeof(float));
    } else if (m_device == DeviceType::CUDA && other.m_device == DeviceType::CPU) {
        CUDABackend::copyHostToDevice(m_data, other.m_data, m_size);
    } else if (m_device == DeviceType::CPU && other.m_device == DeviceType::CUDA) {
        CUDABackend::copyDeviceToHost(m_data, other.m_data, m_size);
    } else {
        CUDABackend::copyDeviceToDevice(m_data, other.m_data, m_size);
    }
}

void Tensor::copyFromHost(const float* hostData, size_t count) {
    if (count > m_size) {
        throw std::out_of_range("Host data count exceeds tensor size");
    }
    if (m_device == DeviceType::CPU) {
        std::memcpy(m_data, hostData, count * sizeof(float));
    } else {
        CUDABackend::copyHostToDevice(m_data, hostData, count);
    }
}

void Tensor::copyToHost(float* hostData, size_t count) const {
    if (count > m_size) {
        throw std::out_of_range("Requested count exceeds tensor size");
    }
    if (m_device == DeviceType::CPU) {
        std::memcpy(hostData, m_data, count * sizeof(float));
    } else {
        CUDABackend::copyDeviceToHost(hostData, m_data, count);
    }
}

std::vector<float> Tensor::toHostVector() const {
    std::vector<float> vec(m_size);
    copyToHost(vec.data(), m_size);
    return vec;
}

Tensor Tensor::reshape(const std::vector<size_t>& newShape) const {
    size_t newSize = 1;
    for (size_t s : newShape) {
        newSize *= s;
    }
    if (newSize != m_size) {
        throw std::invalid_argument("Reshape total elements must match current tensor size");
    }
    Tensor res(*this);
    res.m_shape = newShape;
    res.computeStrides();
    return res;
}

void Tensor::print(const std::string& name, size_t maxElements) const {
    std::cout << "Tensor ";
    if (!name.empty()) std::cout << "'" << name << "' ";
    std::cout << "[";
    for (size_t i = 0; i < m_shape.size(); ++i) {
        std::cout << m_shape[i] << (i + 1 < m_shape.size() ? ", " : "");
    }
    std::cout << "] Device: " << (m_device == DeviceType::CPU ? "CPU" : "CUDA") << " (total elements: " << m_size << "):" << std::endl;

    std::vector<float> vals = toHostVector();
    size_t printCount = std::min(m_size, maxElements);
    std::cout << "  [ ";
    for (size_t i = 0; i < printCount; ++i) {
        std::cout << vals[i] << (i + 1 < printCount ? ", " : "");
    }
    if (m_size > maxElements) {
        std::cout << ", ... (" << (m_size - maxElements) << " more)";
    }
    std::cout << " ]" << std::endl;
}

