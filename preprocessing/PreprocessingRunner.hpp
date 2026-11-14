#pragma once

#include "../data/Dataset.hpp"
#include "../data/ImageReader.hpp"
#include "../data/LabelMapper.hpp"
#include "ImageInspector.hpp"
#include "Resize.hpp"
#include "Padder.hpp"

#include <string>
#include <vector>
#include <cstdint>

// Configuration for the preprocessing pipeline.
struct PreprocessingConfig {
    int targetWidth           = 224;
    int targetHeight          = 224;
    int desiredChannels       = 3;     // Force RGB (3 channels)
    uint8_t padValue          = 0;     // Padding color value (0 = black)
    size_t maxSamplesPerClass = 0;     // 0 = all; >0 = take balanced subset
    size_t maxTotalSamples    = 0;     // 0 = all; >0 = limit total samples
    int numThreads            = 0;     // 0 = hardware concurrency
};

// Summary metrics of a preprocessing run.
struct PreprocessingSummary {
    int totalDiscoveredCat   = 0;
    int totalDiscoveredDog   = 0;
    int totalAttempted       = 0;
    int successCount         = 0;
    int skippedValidCount    = 0;
    int failedCount          = 0;
    int finalCatCount        = 0;
    int finalDogCount        = 0;
    std::vector<std::string> failedPaths;
    std::vector<std::string> failReasons;
    InspectionStats inspectionStats;
};

// Coordinates the entire offline preprocessing pipeline:
//   Dataset -> ImageInspector -> ImageReader -> RGB -> Resize -> Pad (224x224 uint8) -> Validate -> Save PNG
// Memory policy: processes images incrementally without loading whole dataset into RAM.
// Resumption policy: inspects existing files and labels.csv, skipping valid existing files.
class PreprocessingRunner {
public:
    explicit PreprocessingRunner(const PreprocessingConfig& config = {});

    // Inspect pre-existing processed dataset directory and labels.csv to support safe resumption.
    // Populates validExisting: map of output filename -> label.
    static void inspectExistingProcessed(const std::string& outputDir,
                                         int expectedW, int expectedH, int expectedC,
                                         std::vector<std::pair<std::string, int>>& validExisting);

    // Run the complete dataset preprocessing pipeline on the given datasetRoot,
    // streaming outputs directly to outputDir / "images" and updating outputDir / "labels.csv".
    bool processDataset(const std::string& datasetRoot,
                        const std::string& outputDir,
                        PreprocessingSummary& summary);

    // Process a single image from entry: Decode -> RGB -> Resize -> Pad -> Validate.
    // Output is stored in outImage (uint8, 224x224x3).
    bool processSingle(const DatasetEntry& entry,
                       ImageData& outImage,
                       std::string& errorMsg);

    // Validate dimensions and channels of an image.
    bool validateImage(const ImageData& image, std::string& errorMsg) const;

    // Helper: deterministic filename generator (e.g. "cat_00001.png")
    static std::string makeDeterministicFilename(const std::string& className, size_t index);

    const PreprocessingConfig& config() const { return config_; }

    // Print summary report to stdout.
    static void printSummary(const PreprocessingSummary& summary);

private:
    PreprocessingConfig config_;
    ImageReader   reader_;
    LabelMapper   mapper_;
    Resize        resizer_;
    Padder        padder_;
};
