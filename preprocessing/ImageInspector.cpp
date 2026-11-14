#include "ImageInspector.hpp"
#include "../vendor/stb_image.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>

InspectionStats ImageInspector::inspect(const std::vector<std::string>& filepaths) const {
    InspectionStats stats;
    stats.totalFiles = static_cast<int>(filepaths.size());

    if (filepaths.empty()) {
        return stats;
    }

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    std::atomic<size_t> nextIndex{0};
    std::atomic<int> readableCount{0};
    std::atomic<int> corruptedCount{0};
    std::atomic<long long> sumW{0};
    std::atomic<long long> sumH{0};

    std::mutex minMaxMutex;
    int minW = std::numeric_limits<int>::max();
    int maxW = 0;
    int minH = std::numeric_limits<int>::max();
    int maxH = 0;
    int minC = std::numeric_limits<int>::max();
    int maxC = 0;
    std::vector<std::string> corrupted;

    std::vector<std::thread> workers;
    for (unsigned int t = 0; t < numThreads; ++t) {
        workers.emplace_back([&]() {
            int localMinW = std::numeric_limits<int>::max();
            int localMaxW = 0;
            int localMinH = std::numeric_limits<int>::max();
            int localMaxH = 0;
            int localMinC = std::numeric_limits<int>::max();
            int localMaxC = 0;
            int localReadable = 0;
            int localCorrupted = 0;
            long long localSumW = 0;
            long long localSumH = 0;
            std::vector<std::string> localCorruptedPaths;

            while (true) {
                size_t idx = nextIndex.fetch_add(1);
                if (idx >= filepaths.size()) break;

                const auto& path = filepaths[idx];
                int w = 0, h = 0, c = 0;
                if (stbi_info(path.c_str(), &w, &h, &c) && w > 0 && h > 0 && c > 0) {
                    localReadable++;
                    localMinW = std::min(localMinW, w);
                    localMaxW = std::max(localMaxW, w);
                    localMinH = std::min(localMinH, h);
                    localMaxH = std::max(localMaxH, h);
                    localMinC = std::min(localMinC, c);
                    localMaxC = std::max(localMaxC, c);
                    localSumW += w;
                    localSumH += h;
                } else {
                    localCorrupted++;
                    localCorruptedPaths.push_back(path);
                }
            }

            readableCount.fetch_add(localReadable);
            corruptedCount.fetch_add(localCorrupted);
            sumW.fetch_add(localSumW);
            sumH.fetch_add(localSumH);

            std::lock_guard<std::mutex> lock(minMaxMutex);
            minW = std::min(minW, localMinW);
            maxW = std::max(maxW, localMaxW);
            minH = std::min(minH, localMinH);
            maxH = std::max(maxH, localMaxH);
            minC = std::min(minC, localMinC);
            maxC = std::max(maxC, localMaxC);
            for (const auto& cp : localCorruptedPaths) {
                corrupted.push_back(cp);
            }
        });
    }

    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    stats.readableFiles  = readableCount.load();
    stats.corruptedFiles = corruptedCount.load();
    stats.minWidth       = (stats.readableFiles > 0) ? minW : 0;
    stats.maxWidth       = maxW;
    stats.minHeight      = (stats.readableFiles > 0) ? minH : 0;
    stats.maxHeight      = maxH;
    stats.minChannels    = (stats.readableFiles > 0) ? minC : 0;
    stats.maxChannels    = maxC;
    stats.corruptedPaths = std::move(corrupted);

    if (stats.readableFiles > 0) {
        stats.avgWidth  = static_cast<double>(sumW.load()) / stats.readableFiles;
        stats.avgHeight = static_cast<double>(sumH.load()) / stats.readableFiles;
    }

    return stats;
}

InspectionStats ImageInspector::inspect(const Dataset& dataset) const {
    std::vector<std::string> paths;
    paths.reserve(dataset.size());
    for (const auto& entry : dataset.entries()) {
        paths.push_back(entry.filepath);
    }
    return inspect(paths);
}

std::string ImageInspector::generateReport(const InspectionStats& stats) {
    std::ostringstream oss;
    oss << "========== Image Inspection Report ==========\n";
    oss << "Total files scanned : " << stats.totalFiles      << "\n";
    oss << "Readable            : " << stats.readableFiles   << "\n";
    oss << "Corrupted / skipped : " << stats.corruptedFiles  << "\n";
    oss << "----------------------------------------------\n";
    oss << "Width range         : " << stats.minWidth  << " - " << stats.maxWidth  << " px\n";
    oss << "Height range        : " << stats.minHeight << " - " << stats.maxHeight << " px\n";
    oss << "Average width       : " << std::fixed << std::setprecision(1) << stats.avgWidth  << " px\n";
    oss << "Average height      : " << std::fixed << std::setprecision(1) << stats.avgHeight << " px\n";
    oss << "Channel range       : " << stats.minChannels << " - " << stats.maxChannels << "\n";

    if (!stats.corruptedPaths.empty()) {
        oss << "\nCorrupted files identified (" << stats.corruptedPaths.size() << " total):\n";
        size_t limit = std::min(stats.corruptedPaths.size(), size_t(10));
        for (size_t i = 0; i < limit; ++i) {
            oss << "  [CORRUPTED] " << stats.corruptedPaths[i] << "\n";
        }
        if (stats.corruptedPaths.size() > limit) {
            oss << "  ... and " << (stats.corruptedPaths.size() - limit) << " more.\n";
        }
    }
    oss << "==============================================\n";
    return oss.str();
}

void ImageInspector::printReport(const InspectionStats& stats) {
    std::cout << "\n" << generateReport(stats) << "\n";
}
