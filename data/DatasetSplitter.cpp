#include "DatasetSplitter.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <map>

std::vector<SplitEntry> DatasetSplitter::getOrGenerateSplit(
    const std::string& labelsCsvPath,
    const std::string& manifestCsvPath,
    float trainRatio,
    float valRatio,
    float testRatio,
    unsigned int seed) {

    // 1. Check if manifest already exists
    std::ifstream testFile(manifestCsvPath);
    if (testFile.good()) {
        testFile.close();
        std::cout << "[DatasetSplitter] Found existing split manifest: " << manifestCsvPath
                  << ". Loading directly without regenerating." << std::endl;
        return loadManifest(manifestCsvPath);
    }

    // 2. Read input labels.csv
    std::ifstream inFile(labelsCsvPath);
    if (!inFile.is_open()) {
        throw std::runtime_error("DatasetSplitter: Could not open labels CSV file: " + labelsCsvPath);
    }

    std::vector<std::pair<std::string, int>> samples;
    std::string line;
    bool isHeader = true;

    while (std::getline(inFile, line)) {
        if (line.empty()) continue;
        // Strip carriage return if CRLF
        if (line.back() == '\r') line.pop_back();

        if (isHeader) {
            isHeader = false;
            // Skip header if it starts with "filename"
            if (line.find("filename") != std::string::npos) {
                continue;
            }
        }

        std::stringstream ss(line);
        std::string filename, labelStr;
        if (std::getline(ss, filename, ',') && std::getline(ss, labelStr, ',')) {
            try {
                int lbl = std::stoi(labelStr);
                samples.emplace_back(filename, lbl);
            } catch (...) {
                // Ignore malformed rows
            }
        }
    }
    inFile.close();

    if (samples.empty()) {
        throw std::runtime_error("DatasetSplitter: No valid samples read from: " + labelsCsvPath);
    }

    std::cout << "[DatasetSplitter] Generating stratified split for " << samples.size()
              << " samples with seed " << seed << "..." << std::endl;

    // 3. Generate stratified split
    std::vector<SplitEntry> manifest = generateStratifiedSplit(
        samples, trainRatio, valRatio, testRatio, seed
    );

    // 4. Save to disk
    saveManifest(manifestCsvPath, manifest);
    std::cout << "[DatasetSplitter] Saved split manifest to: " << manifestCsvPath << std::endl;

    return manifest;
}

std::vector<SplitEntry> DatasetSplitter::loadManifest(const std::string& manifestCsvPath) {
    std::ifstream file(manifestCsvPath);
    if (!file.is_open()) {
        throw std::runtime_error("DatasetSplitter: Failed to open manifest: " + manifestCsvPath);
    }

    std::vector<SplitEntry> entries;
    std::string line;
    bool isHeader = true;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line.back() == '\r') line.pop_back();

        if (isHeader) {
            isHeader = false;
            if (line.find("filename") != std::string::npos) {
                continue;
            }
        }

        std::stringstream ss(line);
        std::string filename, labelStr, splitStr;
        if (std::getline(ss, filename, ',') &&
            std::getline(ss, labelStr, ',') &&
            std::getline(ss, splitStr, ',')) {
            SplitEntry entry;
            entry.filename = filename;
            entry.label = std::stoi(labelStr);
            entry.split = splitStr;
            entries.push_back(entry);
        }
    }

    return entries;
}

void DatasetSplitter::saveManifest(const std::string& manifestCsvPath, const std::vector<SplitEntry>& entries) {
    std::ofstream out(manifestCsvPath);
    if (!out.is_open()) {
        throw std::runtime_error("DatasetSplitter: Failed to write manifest: " + manifestCsvPath);
    }

    out << "filename,label,split\n";
    for (const auto& e : entries) {
        out << e.filename << "," << e.label << "," << e.split << "\n";
    }
    out.close();
}

std::vector<SplitEntry> DatasetSplitter::generateStratifiedSplit(
    const std::vector<std::pair<std::string, int>>& samples,
    float trainRatio,
    float valRatio,
    float testRatio,
    unsigned int seed) {

    float sumRatio = trainRatio + valRatio + testRatio;
    if (std::abs(sumRatio - 1.0f) > 1e-4f) {
        throw std::invalid_argument("Split ratios must sum to 1.0");
    }

    // Group samples by class label
    std::map<int, std::vector<std::string>> classSamples;
    for (const auto& s : samples) {
        classSamples[s.second].push_back(s.first);
    }

    std::vector<SplitEntry> result;
    result.reserve(samples.size());

    for (auto& pair : classSamples) {
        int label = pair.first;
        auto& files = pair.second;

        // Sort first to ensure deterministic ordering before shuffling across platforms
        std::sort(files.begin(), files.end());

        // Shuffle with seed
        std::mt19937 rng(seed);
        std::shuffle(files.begin(), files.end(), rng);

        size_t totalClass = files.size();
        size_t nTrain = static_cast<size_t>(std::round(static_cast<float>(totalClass) * trainRatio));
        size_t nVal = static_cast<size_t>(std::round(static_cast<float>(totalClass) * valRatio));

        for (size_t i = 0; i < totalClass; ++i) {
            SplitEntry entry;
            entry.filename = files[i];
            entry.label = label;

            if (i < nTrain) {
                entry.split = "train";
            } else if (i < nTrain + nVal) {
                entry.split = "validation";
            } else {
                entry.split = "test";
            }
            result.push_back(entry);
        }
    }

    // Sort final entries by filename for clean human-readable manifest
    std::sort(result.begin(), result.end(), [](const SplitEntry& a, const SplitEntry& b) {
        return a.filename < b.filename;
    });

    return result;
}

std::vector<SplitEntry> DatasetSplitter::filterBySplit(
    const std::vector<SplitEntry>& entries,
    const std::string& splitName) {

    std::vector<SplitEntry> filtered;
    for (const auto& e : entries) {
        if (e.split == splitName) {
            filtered.push_back(e);
        } else if ((splitName == "val" || splitName == "validation") &&
                   (e.split == "val" || e.split == "validation")) {
            filtered.push_back(e);
        }
    }
    return filtered;
}
