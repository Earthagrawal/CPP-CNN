#include "Padder.hpp"
#include <cstring>
#include <stdexcept>
#include <string>

Padder::Padder(int targetWidth, int targetHeight, uint8_t padValue)
    : targetWidth_(targetWidth), targetHeight_(targetHeight), padValue_(padValue) {}

ImageData Padder::apply(const ImageData& input) const {
    if (input.empty()) {
        return {};
    }

    if (input.width > targetWidth_ || input.height > targetHeight_) {
        throw std::invalid_argument(
            "Padder error: input dimensions (" + std::to_string(input.width) + "x" +
            std::to_string(input.height) + ") exceed target canvas (" +
            std::to_string(targetWidth_) + "x" + std::to_string(targetHeight_) + ")");
    }

    ImageData output;
    output.width    = targetWidth_;
    output.height   = targetHeight_;
    output.channels = input.channels;

    size_t totalBytes = static_cast<size_t>(targetWidth_) * targetHeight_ * input.channels;
    output.pixels.resize(totalBytes, padValue_);

    // Symmetrical centered padding
    // For odd remaining difference, integer division gives padLeft;
    // padRight takes the remainder: (targetWidth - input.width - padLeft)
    int padLeft = (targetWidth_  - input.width)  / 2;
    int padTop  = (targetHeight_ - input.height) / 2;

    size_t srcRowBytes = static_cast<size_t>(input.width) * input.channels;
    size_t dstRowBytes = static_cast<size_t>(targetWidth_) * input.channels;

    for (int y = 0; y < input.height; ++y) {
        const uint8_t* srcRow = input.pixels.data() + y * srcRowBytes;
        uint8_t* dstRow = output.pixels.data()
                        + (padTop + y) * dstRowBytes
                        + padLeft * input.channels;
        std::memcpy(dstRow, srcRow, srcRowBytes);
    }

    return output;
}
