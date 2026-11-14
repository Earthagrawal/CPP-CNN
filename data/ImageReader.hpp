#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Holds raw decoded pixel data and metadata for a single image in uint8 format.
// Channels: 1=Gray, 3=RGB, 4=RGBA
// Layout: HWC (Row-major, interleaved channels: [y][x][channel])
struct ImageData {
    std::vector<uint8_t> pixels;   // Raw uint8 bytes in [0, 255]
    int width    = 0;
    int height   = 0;
    int channels = 0;              // e.g. 3 for RGB

    bool empty() const { return pixels.empty(); }
    size_t sizeBytes() const { return pixels.size(); }
};

// Holds normalized floating-point image for training-time ingestion.
// Pixel values in float32 [0.0f, 1.0f].
// Layout: HWC (Row-major, interleaved channels: [y][x][channel])
struct NormalizedImage {
    std::vector<float> pixels;     // float32 values in [0.0f, 1.0f]
    int width    = 0;
    int height   = 0;
    int channels = 0;

    bool empty() const { return pixels.empty(); }
    size_t size() const { return pixels.size(); }
    size_t sizeBytes() const { return pixels.size() * sizeof(float); }
};

// Decodes image files (JPEG, PNG, etc.) using STB image library.
// Also provides conversion to normalized float32 for training time.
class ImageReader {
public:
    // Desired output channels: 0 = keep original, 1 = gray, 3 = RGB, 4 = RGBA.
    explicit ImageReader(int desiredChannels = 3);

    // Decode an image file from disk into uint8 pixel data.
    // Error description is placed in errorMsg if decoding fails.
    ImageData read(const std::string& filepath, std::string& errorMsg) const;

    // Quick header inspection without decoding pixel buffers.
    bool canRead(const std::string& filepath) const;
    bool getInfo(const std::string& filepath, int& width, int& height, int& channels) const;

    // Training-time conversion:
    // Takes uint8 [0, 255] pixels and produces float32 [0.0f, 1.0f] normalized pixels.
    // Formula: x_float32 = float32(x_uint8) / 255.0f
    // Shape: H x W x C (Row-major interleaved RGB)
    static NormalizedImage toNormalizedFloat(const ImageData& image);

    // Direct training-time helper: decodes PNG directly to normalized float32.
    NormalizedImage readAsNormalizedFloat(const std::string& filepath, std::string& errorMsg) const;

    int desiredChannels() const { return desiredChannels_; }

private:
    int desiredChannels_;
};
