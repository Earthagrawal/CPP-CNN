#include "MaxPool2D.hpp"
#include <stdexcept>
#include <cstring>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

MaxPool2D::MaxPool2D(size_t poolSize, size_t stride, const std::string& name)
    : MaxPool2D(poolSize, poolSize, stride, stride, name) {}

MaxPool2D::MaxPool2D(size_t poolH, size_t poolW, size_t strideH, size_t strideW, const std::string& name)
    : Layer(name),
      m_poolH(poolH),
      m_poolW(poolW),
      m_strideH(strideH),
      m_strideW(strideW),
      m_inputShape({}),
      m_device(DeviceType::CPU),
      m_mask(nullptr),
      m_maskSize(0) {}

MaxPool2D::~MaxPool2D() {
    freeMask();
}

void MaxPool2D::allocateMask(size_t size, DeviceType dev) {
    if (m_mask != nullptr && m_maskSize == size && m_device == dev) {
        return;
    }
    freeMask();
    m_maskSize = size;
    m_device = dev;
    if (dev == DeviceType::CPU) {
        m_mask = new int[size]();
    } else {
#ifdef USE_CUDA
        cudaMalloc(reinterpret_cast<void**>(&m_mask), size * sizeof(int));
#else
        m_mask = new int[size]();
        m_device = DeviceType::CPU;
#endif
    }
}

void MaxPool2D::freeMask() {
    if (m_mask != nullptr) {
        if (m_device == DeviceType::CPU) {
            delete[] m_mask;
        } else {
#ifdef USE_CUDA
            cudaFree(m_mask);
#endif
        }
        m_mask = nullptr;
        m_maskSize = 0;
    }
}

Tensor MaxPool2D::forward(const Tensor& input) {
    if (input.dim() != 4) {
        throw std::invalid_argument("MaxPool2D input must be 4D [Batch, Channels, Height, Width]");
    }
    size_t N = input.shape()[0];
    size_t C = input.shape()[1];
    size_t H = input.shape()[2];
    size_t W = input.shape()[3];

    if (H < m_poolH || W < m_poolW) {
        throw std::invalid_argument("Input dimensions smaller than MaxPool kernel");
    }

    size_t outH = (H - m_poolH) / m_strideH + 1;
    size_t outW = (W - m_poolW) / m_strideW + 1;

    m_inputShape = input.shape();
    allocateMask(N * C * outH * outW, input.device());

    Tensor output({N, C, outH, outW}, input.device());

    CUDABackend::maxpool2dForward(input.data(), output.data(), m_mask,
                                  static_cast<int>(N), static_cast<int>(C),
                                  static_cast<int>(H), static_cast<int>(W),
                                  static_cast<int>(m_poolH), static_cast<int>(m_poolW),
                                  static_cast<int>(m_strideH), static_cast<int>(m_strideW),
                                  input.device());

    return output;
}

Tensor MaxPool2D::backward(const Tensor& gradOutput) {
    if (m_inputShape.empty() || m_mask == nullptr) {
        throw std::runtime_error("MaxPool2D backward called before forward");
    }
    size_t N = m_inputShape[0];
    size_t C = m_inputShape[1];
    size_t H = m_inputShape[2];
    size_t W = m_inputShape[3];

    size_t outH = gradOutput.shape()[2];
    size_t outW = gradOutput.shape()[3];

    Tensor gradInput(m_inputShape, m_device);

    CUDABackend::maxpool2dBackward(gradOutput.data(), m_mask, gradInput.data(),
                                   static_cast<int>(N), static_cast<int>(C),
                                   static_cast<int>(H), static_cast<int>(W),
                                   static_cast<int>(outH), static_cast<int>(outW),
                                   m_device);

    return gradInput;
}

void MaxPool2D::toDevice(DeviceType dev) {
    if (m_device != dev && m_mask != nullptr) {
        int* newMask = nullptr;
        if (dev == DeviceType::CPU) {
            newMask = new int[m_maskSize]();
#ifdef USE_CUDA
            cudaMemcpy(newMask, m_mask, m_maskSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(m_mask);
#endif
        } else {
#ifdef USE_CUDA
            cudaMalloc(reinterpret_cast<void**>(&newMask), m_maskSize * sizeof(int));
            cudaMemcpy(newMask, m_mask, m_maskSize * sizeof(int), cudaMemcpyHostToDevice);
            delete[] m_mask;
#endif
        }
        m_mask = newMask;
        m_device = dev;
    }
}

