#pragma once

#include <string>
#include <vector>
#include <utility>
#include <cstddef>

// Holds one entry: an image file path and its associated label and class name.
struct DatasetEntry {
    std::string filepath;
    int label = -1;              // -1 = unknown / unlabelled
    std::string className;       // e.g. "Cat", "Dog"
};

// Maintains image file paths and their associated integer labels.
class Dataset {
public:
    Dataset();

    // Scan `rootDir` for class subdirectories matching classMappings.
    // e.g. rootDir points to ".../PetImages", containing "Cat/" and "Dog/".
    // Supported extensions: .jpg, .jpeg, .png, .bmp
    void loadFromDirectory(const std::string& rootDir,
                           const std::vector<std::pair<std::string, int>>& classMappings);

    // Manually add an entry.
    void addEntry(const std::string& filepath, int label, const std::string& className);

    // Accessors
    const std::vector<DatasetEntry>& entries() const { return entries_; }
    size_t size() const { return entries_.size(); }
    bool empty() const { return entries_.empty(); }

    const DatasetEntry& operator[](size_t index) const { return entries_[index]; }
    const DatasetEntry& getEntry(size_t index) const { return entries_.at(index); }

    // Clear all entries
    void clear() { entries_.clear(); }

    // Return a subset with the first `count` entries.
    std::vector<DatasetEntry> subset(size_t count) const;

    // Return a balanced subset containing up to `samplesPerClass` entries for each class label.
    std::vector<DatasetEntry> getBalancedSubset(size_t samplesPerClass) const;

    // Check if a file extension represents a supported image format.
    static bool isSupportedImage(const std::string& filename);

private:
    std::vector<DatasetEntry> entries_;
};
