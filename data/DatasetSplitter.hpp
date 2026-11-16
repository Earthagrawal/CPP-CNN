#pragma once

#include <string>
#include <vector>
#include <utility>

struct SplitEntry {
    std::string filename;
    int label = 0;              // 0 = Cat, 1 = Dog
    std::string split;          // "train", "validation", "test"
};

class DatasetSplitter {
public:
    // Generate or load existing stratified split manifest.
    // If manifestCsvPath exists on disk, it is loaded directly without modification.
    // If not, it is generated deterministically using seed 42 and written to manifestCsvPath.
    static std::vector<SplitEntry> getOrGenerateSplit(
        const std::string& labelsCsvPath,
        const std::string& manifestCsvPath,
        float trainRatio = 0.8f,
        float valRatio = 0.1f,
        float testRatio = 0.1f,
        unsigned int seed = 42
    );

    // Load an existing split manifest CSV
    static std::vector<SplitEntry> loadManifest(const std::string& manifestCsvPath);

    // Save split manifest to CSV: "filename,label,split"
    static void saveManifest(const std::string& manifestCsvPath, const std::vector<SplitEntry>& entries);

    // Core deterministic stratified partitioning algorithm
    static std::vector<SplitEntry> generateStratifiedSplit(
        const std::vector<std::pair<std::string, int>>& samples,
        float trainRatio = 0.8f,
        float valRatio = 0.1f,
        float testRatio = 0.1f,
        unsigned int seed = 42
    );

    // Filter a list of SplitEntry records by split name ("train", "validation", "test")
    static std::vector<SplitEntry> filterBySplit(
        const std::vector<SplitEntry>& entries,
        const std::string& splitName
    );
};
