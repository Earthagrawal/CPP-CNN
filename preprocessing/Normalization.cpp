#include "Normalization.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

NormalizedImage Normalization::apply(const ImageData& image) {
    return ImageReader::toNormalizedFloat(image);
}

NormalizedImage Normalization::apply(const uint8_t* pixels, int width, int height, int channels) {
    ImageData img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    if (pixels && width > 0 && height > 0 && channels > 0) {
        img.pixels.assign(pixels, pixels + (static_cast<size_t>(width) * height * channels));
    }
    return ImageReader::toNormalizedFloat(img);
}

bool Normalization::validate(const NormalizedImage& img) {
    if (img.empty()) {
        return false;
    }
    for (float v : img.pixels) {
        if (std::isnan(v) || std::isinf(v) || v < 0.0f || v > 1.0f) {
            return false;
        }
    }
    return true;
}

void Normalization::computeMinMax(const NormalizedImage& img, float& outMin, float& outMax) {
    if (img.empty()) {
        outMin = 0.0f;
        outMax = 0.0f;
        return;
    }
    outMin = std::numeric_limits<float>::max();
    outMax = std::numeric_limits<float>::lowest();
    for (float v : img.pixels) {
        if (v < outMin) outMin = v;
        if (v > outMax) outMax = v;
    }
}
