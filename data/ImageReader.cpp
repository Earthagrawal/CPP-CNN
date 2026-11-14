#include "ImageReader.hpp"

// stb_image declarations
#include "../vendor/stb_image.h"

#include <fstream>

ImageReader::ImageReader(int desiredChannels)
    : desiredChannels_(desiredChannels) {}

ImageData ImageReader::read(const std::string& filepath, std::string& errorMsg) const {
    ImageData img;

    // Fast file existence check
    {
        std::ifstream f(filepath, std::ios::binary);
        if (!f.good()) {
            errorMsg = "File does not exist or cannot be opened: " + filepath;
            return img;
        }
    }

    int w = 0, h = 0, origChannels = 0;
    unsigned char* data = stbi_load(
        filepath.c_str(), &w, &h, &origChannels, desiredChannels_
    );

    if (!data) {
        const char* reason = stbi_failure_reason();
        errorMsg = "Failed to decode image: " + filepath +
                   " (" + (reason ? std::string(reason) : "unknown error") + ")";
        return img;
    }

    int outChannels = (desiredChannels_ != 0) ? desiredChannels_ : origChannels;
    size_t totalBytes = static_cast<size_t>(w) * h * outChannels;

    img.width    = w;
    img.height   = h;
    img.channels = outChannels;
    img.pixels.assign(data, data + totalBytes);

    stbi_image_free(data);
    return img;
}

bool ImageReader::canRead(const std::string& filepath) const {
    int w = 0, h = 0, c = 0;
    return stbi_info(filepath.c_str(), &w, &h, &c) != 0;
}

bool ImageReader::getInfo(const std::string& filepath, int& width, int& height, int& channels) const {
    return stbi_info(filepath.c_str(), &width, &height, &channels) != 0;
}

NormalizedImage ImageReader::toNormalizedFloat(const ImageData& image) {
    NormalizedImage out;
    if (image.empty() || image.width <= 0 || image.height <= 0 || image.channels <= 0) {
        return out;
    }

    out.width    = image.width;
    out.height   = image.height;
    out.channels = image.channels;

    size_t totalElements = static_cast<size_t>(image.width) * image.height * image.channels;
    out.pixels.resize(totalElements);

    constexpr float inv255 = 1.0f / 255.0f;
    for (size_t i = 0; i < totalElements; ++i) {
        out.pixels[i] = static_cast<float>(image.pixels[i]) * inv255;
    }

    return out;
}

NormalizedImage ImageReader::readAsNormalizedFloat(const std::string& filepath, std::string& errorMsg) const {
    ImageData raw = read(filepath, errorMsg);
    if (raw.empty()) {
        return {};
    }
    return toNormalizedFloat(raw);
}
