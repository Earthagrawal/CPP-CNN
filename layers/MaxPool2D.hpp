#pragma once

#include "Layer.hpp"
#include <vector>

class MaxPool2D : public Layer {
public:
    MaxPool2D(size_t poolSize = 2, size_t stride = 2, const std::string& name = "MaxPool2D");
    MaxPool2D(size_t poolH, size_t poolW, size_t strideH, size_t strideW, const std::string& name = "MaxPool2D");
    ~MaxPool2D() override;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    void toDevice(DeviceType dev) override;

    size_t poolH() const { return m_poolH; }
    size_t poolW() const { return m_poolW; }
    size_t strideH() const { return m_strideH; }
    size_t strideW() const { return m_strideW; }

private:
    size_t m_poolH;
    size_t m_poolW;
    size_t m_strideH;
    size_t m_strideW;

    std::vector<size_t> m_inputShape;
    DeviceType m_device;
    int* m_mask; // stores argmax linear index in H*W channel
    size_t m_maskSize;

    void allocateMask(size_t size, DeviceType dev);
    void freeMask();
};

