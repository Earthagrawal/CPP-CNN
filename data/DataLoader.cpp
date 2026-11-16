#include "DataLoader.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <future>

DataLoader::DataLoader(
    const std::vector<SplitEntry>& entries,
    const std::string& imagesDir,
    size_t batchSize,
    bool shuffle,
    DeviceType device,
    unsigned int seed)
    : m_entries(entries),
      m_imagesDir(imagesDir),
      m_batchSize(batchSize),
      m_shuffle(shuffle),
      m_device(device),
      m_seed(seed),
      m_rng(seed),
      m_loadSampleIndex(0),
      m_returnedBatches(0),
      m_imageReader(3),
      m_prefetchRunning(false) {

    if (m_batchSize == 0) {
        throw std::invalid_argument("DataLoader: batchSize must be greater than 0");
    }

    m_indices.resize(m_entries.size());
    for (size_t i = 0; i < m_indices.size(); ++i) {
        m_indices[i] = i;
    }

    reset();
}

DataLoader::DataLoader(
    const std::string& manifestCsvPath,
    const std::string& imagesDir,
    const std::string& splitName,
    size_t batchSize,
    bool shuffle,
    DeviceType device,
    unsigned int seed)
    : DataLoader(
          DatasetSplitter::filterBySplit(DatasetSplitter::loadManifest(manifestCsvPath), splitName),
          imagesDir,
          batchSize,
          shuffle,
          device,
          seed) {
}

DataLoader::~DataLoader() {
    if (m_prefetchRunning && m_prefetchFuture.valid()) {
        try {
            m_prefetchFuture.wait();
        } catch (...) {
        }
    }
}

void DataLoader::reset() {
    if (m_prefetchRunning && m_prefetchFuture.valid()) {
        try {
            m_prefetchFuture.wait();
        } catch (...) {
        }
    }
    m_prefetchRunning = false;

    m_loadSampleIndex = 0;
    m_returnedBatches = 0;

    if (m_shuffle && !m_indices.empty()) {
        std::shuffle(m_indices.begin(), m_indices.end(), m_rng);
    }

    // Prefetch first batch asynchronously
    if (m_loadSampleIndex < m_indices.size()) {
        m_prefetchFuture = std::async(std::launch::async, [this]() {
            return loadNextBatchSync();
        });
        m_prefetchRunning = true;
    }
}

bool DataLoader::hasNext() const {
    return m_returnedBatches < numBatches();
}

size_t DataLoader::numBatches() const {
    if (m_entries.empty() || m_batchSize == 0) return 0;
    return (m_entries.size() + m_batchSize - 1) / m_batchSize;
}

Batch DataLoader::loadNextBatchSync() {
    size_t remaining = m_indices.size() - m_loadSampleIndex;
    size_t currentB = std::min(m_batchSize, remaining);

    const size_t C = 3;
    const size_t H = 224;
    const size_t W = 224;
    const size_t imgPixels = C * H * W;

    std::vector<float> h_images(currentB * imgPixels, 0.0f);
    std::vector<float> h_labels(currentB, 0.0f);
    std::vector<std::string> filenames(currentB);

    constexpr float inv255 = 1.0f / 255.0f;

    // Multi-threaded decoding: spawn worker threads across available CPU cores
    unsigned int numThreads = std::min(8u, static_cast<unsigned int>(std::max(1ULL, static_cast<unsigned long long>(std::thread::hardware_concurrency()))));
    numThreads = std::min(numThreads, static_cast<unsigned int>(currentB));

    std::vector<std::thread> workers;
    workers.reserve(numThreads);
    std::vector<std::string> threadErrors(currentB);

    size_t baseIdx = m_loadSampleIndex;

    for (unsigned int t = 0; t < numThreads; ++t) {
        workers.emplace_back([&, t]() {
            ImageReader localReader(3);
            for (size_t b = t; b < currentB; b += numThreads) {
                size_t sampleIdx = m_indices[baseIdx + b];
                const SplitEntry& entry = m_entries[sampleIdx];
                filenames[b] = entry.filename;
                h_labels[b] = static_cast<float>(entry.label);

                std::string fullPath = m_imagesDir + "/" + entry.filename;
                std::string errMsg;
                ImageData raw = localReader.read(fullPath, errMsg);

                if (raw.empty() || raw.width != static_cast<int>(W) || raw.height != static_cast<int>(H) || raw.channels != static_cast<int>(C)) {
                    threadErrors[b] = "Failed to load image or dimension mismatch: " + fullPath + " (" + errMsg + ")";
                    return;
                }

                const uint8_t* pRaw = raw.pixels.data();
                for (size_t c = 0; c < C; ++c) {
                    float* dstChannel = h_images.data() + (b * C + c) * (H * W);
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            size_t hwcIdx = (h * W + w) * C + c;
                            size_t hwIdx  = h * W + w;
                            dstChannel[hwIdx] = static_cast<float>(pRaw[hwcIdx]) * inv255;
                        }
                    }
                }
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    for (size_t b = 0; b < currentB; ++b) {
        if (!threadErrors[b].empty()) {
            throw std::runtime_error("DataLoader: " + threadErrors[b]);
        }
    }

    m_loadSampleIndex += currentB;

    Tensor imgTensor({currentB, C, H, W}, h_images, DeviceType::CPU);
    Tensor lblTensor({currentB, 1}, h_labels, DeviceType::CPU);

    Batch batch;
    batch.filenames = std::move(filenames);

    if (m_device == DeviceType::CUDA) {
        batch.images = imgTensor.to(DeviceType::CUDA);
        batch.labels = lblTensor.to(DeviceType::CUDA);
    } else {
        batch.images = std::move(imgTensor);
        batch.labels = std::move(lblTensor);
    }

    return batch;
}

Batch DataLoader::nextBatch() {
    if (!hasNext()) {
        throw std::out_of_range("DataLoader: No more batches available in this epoch");
    }

    Batch currentBatch;
    if (m_prefetchRunning && m_prefetchFuture.valid()) {
        currentBatch = m_prefetchFuture.get();
        m_prefetchRunning = false;
    } else {
        currentBatch = loadNextBatchSync();
    }

    m_returnedBatches++;

    // Launch background prefetch for the NEXT batch if more remain
    if (m_loadSampleIndex < m_indices.size()) {
        m_prefetchFuture = std::async(std::launch::async, [this]() {
            return loadNextBatchSync();
        });
        m_prefetchRunning = true;
    }

    return currentBatch;
}
