#pragma once

#include "../data/ImageReader.hpp"

// Resizes an image to fit within targetWidth x targetHeight
// while strictly preserving original aspect ratio.
//
// Scaling factor formula:
//   s = min(target_width / original_width, target_height / original_height)
//   new_width = round(original_width * s)
//   new_height = round(original_height * s)
//
// Never stretches or crops images. Uses bilinear interpolation.
class Resize {
public:
    Resize(int targetWidth, int targetHeight);

    // Returns a new ImageData containing the resized pixel data.
    ImageData apply(const ImageData& input) const;

    int targetWidth()  const { return targetWidth_;  }
    int targetHeight() const { return targetHeight_; }

private:
    int targetWidth_;
    int targetHeight_;

    // Bilinear interpolation sampling at continuous (srcX, srcY) coordinates.
    static void sampleBilinear(const uint8_t* src, int srcW, int srcH, int channels,
                               float srcX, float srcY, uint8_t* outPixel);
};
