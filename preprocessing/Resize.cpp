#include "Resize.hpp"
#include <algorithm>
#include <cmath>

Resize::Resize(int targetWidth, int targetHeight)
    : targetWidth_(targetWidth), targetHeight_(targetHeight) {}

ImageData Resize::apply(const ImageData& input) const {
    if (input.empty() || input.width <= 0 || input.height <= 0) {
        return {};
    }

    // Formula:
    // s = min(target_width / original_width, target_height / original_height)
    float scaleX = static_cast<float>(targetWidth_)  / static_cast<float>(input.width);
    float scaleY = static_cast<float>(targetHeight_) / static_cast<float>(input.height);
    float scale  = std::min(scaleX, scaleY);

    // new_width = round(original_width * s)
    // new_height = round(original_height * s)
    int newW = static_cast<int>(std::round(input.width  * scale));
    int newH = static_cast<int>(std::round(input.height * scale));

    // Clamp within valid target boundaries
    newW = std::clamp(newW, 1, targetWidth_);
    newH = std::clamp(newH, 1, targetHeight_);

    ImageData output;
    output.width    = newW;
    output.height   = newH;
    output.channels = input.channels;
    output.pixels.resize(static_cast<size_t>(newW) * newH * input.channels);

    const uint8_t* src = input.pixels.data();

    for (int y = 0; y < newH; ++y) {
        // Map center of destination pixel to continuous source coordinate
        float srcY = (static_cast<float>(y) + 0.5f) * static_cast<float>(input.height) / static_cast<float>(newH) - 0.5f;

        for (int x = 0; x < newW; ++x) {
            float srcX = (static_cast<float>(x) + 0.5f) * static_cast<float>(input.width) / static_cast<float>(newW) - 0.5f;

            size_t dstIdx = (static_cast<size_t>(y) * newW + x) * input.channels;
            sampleBilinear(src, input.width, input.height, input.channels,
                           srcX, srcY, &output.pixels[dstIdx]);
        }
    }

    return output;
}

void Resize::sampleBilinear(const uint8_t* src, int srcW, int srcH, int channels,
                            float srcX, float srcY, uint8_t* outPixel) {
    // Clamp to valid sample domain
    srcX = std::max(0.0f, std::min(srcX, static_cast<float>(srcW - 1)));
    srcY = std::max(0.0f, std::min(srcY, static_cast<float>(srcH - 1)));

    int x0 = static_cast<int>(std::floor(srcX));
    int y0 = static_cast<int>(std::floor(srcY));
    int x1 = std::min(x0 + 1, srcW - 1);
    int y1 = std::min(y0 + 1, srcH - 1);

    float xFrac = srcX - static_cast<float>(x0);
    float yFrac = srcY - static_cast<float>(y0);

    for (int c = 0; c < channels; ++c) {
        float p00 = static_cast<float>(src[(y0 * srcW + x0) * channels + c]);
        float p10 = static_cast<float>(src[(y0 * srcW + x1) * channels + c]);
        float p01 = static_cast<float>(src[(y1 * srcW + x0) * channels + c]);
        float p11 = static_cast<float>(src[(y1 * srcW + x1) * channels + c]);

        float top    = p00 * (1.0f - xFrac) + p10 * xFrac;
        float bottom = p01 * (1.0f - xFrac) + p11 * xFrac;
        float val    = top * (1.0f - yFrac) + bottom * yFrac;

        outPixel[c] = static_cast<uint8_t>(std::clamp(std::round(val), 0.0f, 255.0f));
    }
}
