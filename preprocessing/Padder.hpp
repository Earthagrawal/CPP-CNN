#pragma once

#include "../data/ImageReader.hpp"
#include <cstdint>

// Adds centered padding to an image to achieve exact target dimensions (e.g. 224x224).
// Handles odd padding amounts symmetrically, placing any single remainder pixel
// on the right or bottom border.
class Padder {
public:
    // `padValue`: pixel value used for padding canvas (0 = black).
    Padder(int targetWidth, int targetHeight, uint8_t padValue = 0);

    // Returns a new ImageData padded to exactly targetWidth x targetHeight.
    // Throws std::invalid_argument if input dimensions exceed target dimensions.
    ImageData apply(const ImageData& input) const;

    int     targetWidth()  const { return targetWidth_;  }
    int     targetHeight() const { return targetHeight_; }
    uint8_t padValue()     const { return padValue_;     }

private:
    int     targetWidth_;
    int     targetHeight_;
    uint8_t padValue_;
};
