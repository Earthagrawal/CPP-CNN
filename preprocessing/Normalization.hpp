#pragma once

#include "../data/ImageReader.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>

// Helper class for uint8 [0, 255] to float32 [0.0f, 1.0f] conversion and validation.
// Note: Offline dataset storage is uint8 PNG. Normalization is applied at training time.
class Normalization {
public:
    Normalization() = default;

    // Convert uint8 image to normalized float32 [0.0f, 1.0f]
    static NormalizedImage apply(const ImageData& image);
    static NormalizedImage apply(const uint8_t* pixels, int width, int height, int channels);

    // Verifies that all pixel values are within [0.0f, 1.0f] and free of NaN/Inf.
    static bool validate(const NormalizedImage& img);

    // Compute min and max pixel values in a normalized image.
    static void computeMinMax(const NormalizedImage& img, float& outMin, float& outMax);
};
