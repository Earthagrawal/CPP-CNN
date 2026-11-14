#include "PreprocessingRunner.hpp"
#include "../vendor/stb_image_write.h"
#include "../vendor/stb_image.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <atomic>

namespace fs = std::filesystem;

PreprocessingRunner::PreprocessingRunner(const PreprocessingConfig& config)
    : config_(config)
    , reader_(config.desiredChannels)
    , resizer_(config.targetWidth, config.targetHeight)
    , padder_(config.targetWidth, config.targetHeight, config.padValue)
{}

std::string PreprocessingRunner::makeDeterministicFilename(
    const std::string& className, size_t index)
{
    std::string lowerClass = className;
    std::transform(lowerClass.begin(), lowerClass.end(), lowerClass.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    std::ostringstream oss;
    oss << lowerClass << "_" << std::setw(5) << std::setfill('0') << index << ".png";
    return oss.str();
}

bool PreprocessingRunner::validateImage(const ImageData& image, std::string& errorMsg) const {
    if (image.empty()) {
        errorMsg = "Empty image buffer";
        return false;
    }
    if (image.width != config_.targetWidth || image.height != config_.targetHeight) {
        errorMsg = "Dimension mismatch: got " + std::to_string(image.width) + "x" +
                   std::to_string(image.height) + ", expected " +
                   std::to_string(config_.targetWidth) + "x" +
                   std::to_string(config_.targetHeight);
        return false;
    }
    if (image.channels != config_.desiredChannels) {
        errorMsg = "Channel mismatch: got " + std::to_string(image.channels) +
                   ", expected " + std::to_string(config_.desiredChannels);
        return false;
    }
    return true;
}

bool PreprocessingRunner::processSingle(const DatasetEntry& entry,
                                        ImageData& outImage,
                                        std::string& errorMsg)
{
    // 1. Read raw image and enforce RGB channels
    ImageData raw = reader_.read(entry.filepath, errorMsg);
    if (raw.empty()) {
        return false;
    }

    // 2. Aspect-ratio preserving Resize
    ImageData resized = resizer_.apply(raw);
    if (resized.empty()) {
        errorMsg = "Resize failed on: " + entry.filepath;
        return false;
    }

    // 3. Centered Padder to exact target dimensions (224x224)
    try {
        outImage = padder_.apply(resized);
    } catch (const std::exception& e) {
        errorMsg = std::string("Padding failed: ") + e.what();
        return false;
    }

    // 4. Validate output dimensions and channels
    if (!validateImage(outImage, errorMsg)) {
        return false;
    }

    return true;
}

void PreprocessingRunner::inspectExistingProcessed(
    const std::string& outputDir,
    int expectedW, int expectedH, int expectedC,
    std::vector<std::pair<std::string, int>>& validExisting)
{
    validExisting.clear();
    fs::path imagesDir = fs::path(outputDir) / "images";
    fs::path csvPath   = fs::path(outputDir) / "labels.csv";

    if (!fs::exists(imagesDir) || !fs::exists(csvPath)) {
        return;
    }

    // Read existing labels.csv
    std::ifstream csv(csvPath.string());
    if (!csv.is_open()) {
        return;
    }

    std::string line;
    // Read header
    if (!std::getline(csv, line)) {
        return;
    }

    std::unordered_set<std::string> seenFiles;

    while (std::getline(csv, line)) {
        if (line.empty()) continue;
        auto comma = line.find(',');
        if (comma == std::string::npos) continue;

        std::string filename = line.substr(0, comma);
        // Trim whitespace
        while (!filename.empty() && std::isspace(static_cast<unsigned char>(filename.back()))) {
            filename.pop_back();
        }
        int label = -1;
        try {
            label = std::stoi(line.substr(comma + 1));
        } catch (...) {
            continue;
        }

        if (seenFiles.count(filename)) {
            continue; // avoid duplicate entries in CSV
        }

        // Verify that the file actually exists on disk and has correct dimensions
        fs::path imgPath = imagesDir / filename;
        if (fs::exists(imgPath)) {
            int w = 0, h = 0, c = 0;
            if (stbi_info(imgPath.string().c_str(), &w, &h, &c) &&
                w == expectedW && h == expectedH && c == expectedC) {
                validExisting.push_back({filename, label});
                seenFiles.insert(filename);
            }
        }
    }
}

bool PreprocessingRunner::processDataset(const std::string& datasetRoot,
                                         const std::string& outputDir,
                                         PreprocessingSummary& summary)
{
    fs::path outPath(outputDir);
    fs::path imagesDir = outPath / "images";
    fs::create_directories(imagesDir);

    // 1. Scan dataset files
    std::cout << "[Pipeline] Scanning raw dataset from " << datasetRoot << "...\n";
    Dataset dataset;
    std::vector<std::pair<std::string, int>> classMappings;
    for (const auto& [name, label] : mapper_.mappings()) {
        classMappings.push_back({name, label});
    }
    dataset.loadFromDirectory(datasetRoot, classMappings);

    // Record discovered counts
    for (const auto& entry : dataset.entries()) {
        if (entry.label == 0) summary.totalDiscoveredCat++;
        if (entry.label == 1) summary.totalDiscoveredDog++;
    }

    std::vector<DatasetEntry> entries;
    if (config_.maxSamplesPerClass > 0) {
        entries = dataset.getBalancedSubset(config_.maxSamplesPerClass);
    } else if (config_.maxTotalSamples > 0 && config_.maxTotalSamples < dataset.size()) {
        entries = dataset.subset(config_.maxTotalSamples);
    } else {
        entries = dataset.entries();
    }

    summary.totalAttempted = static_cast<int>(entries.size());

    // 2. Pre-inspection
    std::cout << "[Pipeline] Inspecting " << entries.size() << " dataset entries...\n";
    std::vector<std::string> paths;
    paths.reserve(entries.size());
    for (const auto& e : entries) {
        paths.push_back(e.filepath);
    }

    ImageInspector inspector;
    summary.inspectionStats = inspector.inspect(paths);
    ImageInspector::printReport(summary.inspectionStats);

    // 3. Inspect existing processed files for safe resumption
    std::vector<std::pair<std::string, int>> validExisting;
    inspectExistingProcessed(outputDir, config_.targetWidth, config_.targetHeight, config_.desiredChannels, validExisting);
    std::unordered_map<std::string, int> existingMap;
    for (const auto& item : validExisting) {
        existingMap[item.first] = item.second;
    }
    if (!existingMap.empty()) {
        std::cout << "[Resumption] Found " << existingMap.size()
                  << " already valid processed PNG images. They will be preserved and skipped.\n";
    }

    // 4. Organize work per class so deterministic naming is sequential:
    // e.g. cat_00001.png, cat_00002.png...
    // dog_00001.png, dog_00002.png...
    std::unordered_map<std::string, std::vector<DatasetEntry>> classEntries;
    for (const auto& e : entries) {
        classEntries[e.className].push_back(e);
    }

    // Prepare container for final manifest records
    struct ManifestRecord {
        std::string filename;
        int label;
    };
    std::vector<ManifestRecord> finalManifest;
    std::mutex manifestMutex;

    // Determine thread count
    unsigned int threadsToUse = config_.numThreads > 0 ?
        config_.numThreads : std::thread::hardware_concurrency();
    if (threadsToUse == 0) threadsToUse = 4;

    std::cout << "[Pipeline] Processing images incrementally using "
              << threadsToUse << " worker threads...\n";

    // Global counters for progress
    std::atomic<int> processedCount{0};
    std::atomic<int> successCount{0};
    std::atomic<int> skippedCount{0};
    std::atomic<int> failedCount{0};
    std::atomic<int> finalCat{0};
    std::atomic<int> finalDog{0};

    std::mutex failMutex;

    // Process each class
    for (auto& [className, classList] : classEntries) {
        std::cout << "\n[Pipeline] Processing class " << className
                  << " (" << classList.size() << " images)...\n";

        // Assign deterministic filename to each item in classList
        struct TaskItem {
            DatasetEntry entry;
            std::string outFilename;
        };
        std::vector<TaskItem> tasks;
        tasks.reserve(classList.size());

        for (size_t i = 0; i < classList.size(); ++i) {
            std::string fname = makeDeterministicFilename(className, i + 1);
            tasks.push_back({classList[i], fname});
        }

        std::atomic<size_t> taskIndex{0};
        std::vector<std::thread> workers;

        for (unsigned int t = 0; t < threadsToUse; ++t) {
            workers.emplace_back([&, className]() {
                ImageReader localReader(config_.desiredChannels);
                Resize localResize(config_.targetWidth, config_.targetHeight);
                Padder localPadder(config_.targetWidth, config_.targetHeight, config_.padValue);

                while (true) {
                    size_t idx = taskIndex.fetch_add(1);
                    if (idx >= tasks.size()) break;

                    const auto& task = tasks[idx];
                    fs::path outFilePath = imagesDir / task.outFilename;

                    // Check if already valid from safe resumption
                    bool alreadyValid = false;
                    auto it = existingMap.find(task.outFilename);
                    if (it != existingMap.end() && it->second == task.entry.label) {
                        // Double check file on disk
                        int w = 0, h = 0, c = 0;
                        if (stbi_info(outFilePath.string().c_str(), &w, &h, &c) &&
                            w == config_.targetWidth && h == config_.targetHeight && c == config_.desiredChannels) {
                            alreadyValid = true;
                        }
                    }

                    if (alreadyValid) {
                        skippedCount.fetch_add(1);
                        successCount.fetch_add(1);
                        if (task.entry.label == 0) finalCat.fetch_add(1);
                        if (task.entry.label == 1) finalDog.fetch_add(1);

                        std::lock_guard<std::mutex> lock(manifestMutex);
                        finalManifest.push_back({task.outFilename, task.entry.label});
                    } else {
                        // Process single image
                        std::string errorMsg;
                        ImageData raw = localReader.read(task.entry.filepath, errorMsg);
                        if (raw.empty()) {
                            failedCount.fetch_add(1);
                            std::lock_guard<std::mutex> lock(failMutex);
                            summary.failedPaths.push_back(task.entry.filepath);
                            summary.failReasons.push_back(errorMsg);
                        } else {
                            ImageData resized = localResize.apply(raw);
                            // raw is released here
                            raw.pixels.clear();
                            raw.pixels.shrink_to_fit();

                            if (resized.empty()) {
                                failedCount.fetch_add(1);
                                std::lock_guard<std::mutex> lock(failMutex);
                                summary.failedPaths.push_back(task.entry.filepath);
                                summary.failReasons.push_back("Resize failed");
                            } else {
                                ImageData padded;
                                try {
                                    padded = localPadder.apply(resized);
                                } catch (const std::exception& e) {
                                    failedCount.fetch_add(1);
                                    std::lock_guard<std::mutex> lock(failMutex);
                                    summary.failedPaths.push_back(task.entry.filepath);
                                    summary.failReasons.push_back(e.what());
                                    continue;
                                }
                                // resized is released here
                                resized.pixels.clear();
                                resized.pixels.shrink_to_fit();

                                // Validate dimensions
                                if (padded.width != config_.targetWidth ||
                                    padded.height != config_.targetHeight ||
                                    padded.channels != config_.desiredChannels) {
                                    failedCount.fetch_add(1);
                                    std::lock_guard<std::mutex> lock(failMutex);
                                    summary.failedPaths.push_back(task.entry.filepath);
                                    summary.failReasons.push_back("Dimension validation failed");
                                } else {
                                    // Save as PNG (uint8)
                                    int stride = padded.width * padded.channels;
                                    int res = stbi_write_png(outFilePath.string().c_str(),
                                                             padded.width, padded.height,
                                                             padded.channels, padded.pixels.data(),
                                                             stride);
                                    // padded is released
                                    padded.pixels.clear();
                                    padded.pixels.shrink_to_fit();

                                    if (!res) {
                                        failedCount.fetch_add(1);
                                        std::lock_guard<std::mutex> lock(failMutex);
                                        summary.failedPaths.push_back(task.entry.filepath);
                                        summary.failReasons.push_back("Failed to write PNG file");
                                    } else {
                                        successCount.fetch_add(1);
                                        if (task.entry.label == 0) finalCat.fetch_add(1);
                                        if (task.entry.label == 1) finalDog.fetch_add(1);

                                        std::lock_guard<std::mutex> lock(manifestMutex);
                                        finalManifest.push_back({task.outFilename, task.entry.label});
                                    }
                                }
                            }
                        }
                    }

                    int totalDone = processedCount.fetch_add(1) + 1;
                    if (totalDone % 250 == 0 || totalDone == summary.totalAttempted) {
                        std::cout << "\r  Progress: [" << totalDone << " / " << summary.totalAttempted
                                  << "]  Success: " << successCount.load()
                                  << " (Skipped: " << skippedCount.load() << ")"
                                  << "  Failed: " << failedCount.load() << std::flush;
                    }
                }
            });
        }

        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
    }

    std::cout << "\n\n[Pipeline] Image processing pass completed. Writing labels.csv...\n";

    // Sort manifest deterministically by filename
    std::sort(finalManifest.begin(), finalManifest.end(),
              [](const ManifestRecord& a, const ManifestRecord& b) {
                  return a.filename < b.filename;
              });

    // Write labels.csv safely using atomic temp file replacement strategy
    fs::path csvPath = outPath / "labels.csv";
    fs::path tempCsvPath = outPath / "labels_tmp.csv";

    {
        std::ofstream csv(tempCsvPath.string(), std::ios::out | std::ios::trunc);
        if (!csv.is_open()) {
            std::cerr << "[ERROR] Could not open temporary labels file for writing: "
                      << tempCsvPath << "\n";
            return false;
        }

        csv << "filename,label\n";
        for (const auto& item : finalManifest) {
            csv << item.filename << "," << item.label << "\n";
        }
        csv.close();
    }

    // Atomic rename
    std::error_code ec;
    fs::rename(tempCsvPath, csvPath, ec);
    if (ec) {
        // Fallback for Windows if destination exists
        fs::remove(csvPath, ec);
        fs::rename(tempCsvPath, csvPath, ec);
    }

    // Fill summary
    summary.successCount      = successCount.load();
    summary.skippedValidCount = skippedCount.load();
    summary.failedCount       = failedCount.load();
    summary.finalCatCount     = finalCat.load();
    summary.finalDogCount     = finalDog.load();

    return true;
}

void PreprocessingRunner::printSummary(const PreprocessingSummary& summary) {
    std::cout << "\n=======================================================\n";
    std::cout << "          Final Preprocessing Summary Report           \n";
    std::cout << "=======================================================\n";
    std::cout << "Total Cat images discovered : " << summary.totalDiscoveredCat << "\n";
    std::cout << "Total Dog images discovered : " << summary.totalDiscoveredDog << "\n";
    std::cout << "Total images attempted      : " << summary.totalAttempted     << "\n";
    std::cout << "Successfully processed      : " << summary.successCount       << "\n";
    std::cout << "  - Already valid & skipped : " << summary.skippedValidCount  << "\n";
    std::cout << "  - Newly processed PNGs    : " << (summary.successCount - summary.skippedValidCount) << "\n";
    std::cout << "Failed / corrupted images   : " << summary.failedCount        << "\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Final Cat output count      : " << summary.finalCatCount      << "\n";
    std::cout << "Final Dog output count      : " << summary.finalDogCount      << "\n";
    std::cout << "Total output image files    : " << (summary.finalCatCount + summary.finalDogCount) << "\n";

    if (!summary.failedPaths.empty()) {
        std::cout << "\nCorrupted or unreadable image log (first 15):\n";
        size_t limit = std::min(summary.failedPaths.size(), size_t(15));
        for (size_t i = 0; i < limit; ++i) {
            std::cout << "  [CORRUPTED] " << summary.failedPaths[i]
                      << "  --> Reason: " << summary.failReasons[i] << "\n";
        }
        if (summary.failedPaths.size() > limit) {
            std::cout << "  ... and " << (summary.failedPaths.size() - limit) << " more.\n";
        }
    }
    std::cout << "=======================================================\n\n";
}
