#pragma once

#include "../core/Tensor.hpp"
#include "DatasetSplitter.hpp"
#include "ImageReader.hpp"
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <future>

struct Batch {
    Tensor images;                      // Shape: [B, 3, 224, 224], FP32 in [0, 1], NCHW
    Tensor labels;                      // Shape: [B, 1], FP32 (0.0f = Cat, 1.0f = Dog)
    std::vector<std::string> filenames; // Associated filenames for debugging/logging

    size_t batchSize() const { return labels.empty() ? 0 : labels.shape()[0]; }
    DeviceType device() const { return images.device(); }
};

class DataLoader {
public:
    DataLoader(
        const std::vector<SplitEntry>& entries,
        const std::string& imagesDir,
        size_t batchSize = 32,
        bool shuffle = false,
        DeviceType device = DeviceType::CPU,
        unsigned int seed = 42
    );

    // Convenience constructor loading directly from manifest file for a specific split
    DataLoader(
        const std::string& manifestCsvPath,
        const std::string& imagesDir,
        const std::string& splitName,
        size_t batchSize = 32,
        bool shuffle = false,
        DeviceType device = DeviceType::CPU,
        unsigned int seed = 42
    );

    ~DataLoader();

    // Iteration controls
    void reset();
    bool hasNext() const;
    Batch nextBatch();

    // Dataset and batch metadata
    size_t numSamples() const { return m_entries.size(); }
    size_t batchSize() const { return m_batchSize; }
    size_t numBatches() const;
    size_t currentBatchIndex() const { return m_returnedBatches; }
    bool isShuffled() const { return m_shuffle; }
    DeviceType device() const { return m_device; }

    const std::vector<SplitEntry>& entries() const { return m_entries; }

private:
    Batch loadNextBatchSync();

    std::vector<SplitEntry> m_entries;
    std::vector<size_t> m_indices;
    std::string m_imagesDir;
    size_t m_batchSize;
    bool m_shuffle;
    DeviceType m_device;
    unsigned int m_seed;
    std::mt19937 m_rng;

    size_t m_loadSampleIndex;
    size_t m_returnedBatches;
    ImageReader m_imageReader;

    std::future<Batch> m_prefetchFuture;
    bool m_prefetchRunning = false;
};
