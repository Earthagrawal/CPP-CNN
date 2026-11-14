#include "Dataset.hpp"

#include <filesystem>
#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace fs = std::filesystem;

Dataset::Dataset() {}

void Dataset::loadFromDirectory(const std::string& rootDir,
                                const std::vector<std::pair<std::string, int>>& classMappings) {
    entries_.clear();

    for (const auto& [className, label] : classMappings) {
        fs::path classDir = fs::path(rootDir) / className;
        if (!fs::is_directory(classDir)) {
            continue;
        }

        // Collect and sort paths deterministically
        std::vector<std::string> fileList;
        for (const auto& entry : fs::directory_iterator(classDir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            std::string filename = entry.path().filename().string();
            if (!isSupportedImage(filename)) {
                continue;
            }
            fileList.push_back(entry.path().string());
        }

        // Natural/alphanumeric sort for deterministic ordering
        std::sort(fileList.begin(), fileList.end());

        for (const auto& path : fileList) {
            DatasetEntry de;
            de.filepath  = path;
            de.label     = label;
            de.className = className;
            entries_.push_back(std::move(de));
        }
    }
}

void Dataset::addEntry(const std::string& filepath, int label, const std::string& className) {
    entries_.push_back({filepath, label, className});
}

std::vector<DatasetEntry> Dataset::subset(size_t count) const {
    size_t n = std::min(count, entries_.size());
    return std::vector<DatasetEntry>(entries_.begin(), entries_.begin() + n);
}

std::vector<DatasetEntry> Dataset::getBalancedSubset(size_t samplesPerClass) const {
    std::unordered_map<int, size_t> classCount;
    std::vector<DatasetEntry> result;

    for (const auto& entry : entries_) {
        size_t current = classCount[entry.label];
        if (current < samplesPerClass) {
            result.push_back(entry);
            classCount[entry.label] = current + 1;
        }
    }

    return result;
}

bool Dataset::isSupportedImage(const std::string& filename) {
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (lower.size() >= 4) {
        if (lower.rfind(".jpg")  == lower.size() - 4 ||
            lower.rfind(".png")  == lower.size() - 4 ||
            lower.rfind(".bmp")  == lower.size() - 4) {
            return true;
        }
    }
    if (lower.size() >= 5) {
        if (lower.rfind(".jpeg") == lower.size() - 5) {
            return true;
        }
    }
    return false;
}
