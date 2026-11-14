// CatDogCNN - Main Entry Point: Complete Dataset Preprocessing (Phase 1)

#include "data/ImageReader.hpp"
#include "data/LabelMapper.hpp"
#include "data/Dataset.hpp"
#include "preprocessing/ImageInspector.hpp"
#include "preprocessing/PreprocessingRunner.hpp"
#include "preprocessing/Normalization.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

static std::string resolvePath(const std::string& candidateA, const std::string& candidateB) {
    if (fs::exists(candidateA)) return candidateA;
    if (fs::exists(candidateB)) return candidateB;
    return candidateA;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================================\n";
    std::cout << "    CatDogCNN - Complete Dataset Preprocessing Pipeline  \n";
    std::cout << "========================================================\n\n";

    // 1. Resolve Dataset Paths
    std::string datasetRoot = resolvePath("../dataset/PetImages", "dataset/PetImages");
    std::string processedDir = resolvePath("../dataset/processed", "dataset/processed");
    size_t maxSamplesPerClass = 0; // 0 = process ENTIRE dataset

    if (argc >= 2) datasetRoot  = argv[1];
    if (argc >= 3) processedDir = argv[2];
    if (argc >= 4) maxSamplesPerClass = std::stoul(argv[3]);

    std::cout << "Configuration:\n";
    std::cout << "  Raw Dataset Root  : " << datasetRoot  << "\n";
    std::cout << "  Processed Storage : " << processedDir << "\n";
    if (maxSamplesPerClass > 0) {
        std::cout << "  Sample Limit      : " << maxSamplesPerClass << " per class\n";
    } else {
        std::cout << "  Scope             : ENTIRE DATASET (All Cat & Dog images)\n";
    }
    std::cout << "  Target Dimension  : 224 x 224 x 3 (RGB)\n";
    std::cout << "  Disk Storage      : Lossless PNG (uint8, [0, 255])\n";
    std::cout << "  Training-time     : float32, [0.0, 1.0] (HWC layout)\n";
    std::cout << "  Safe Resumption   : Enabled (Existing valid PNGs will be preserved)\n\n";

    if (!fs::exists(datasetRoot)) {
        std::cerr << "[ERROR] Dataset directory not found: " << datasetRoot << "\n";
        return 1;
    }

    // 2. Initialize LabelMapper
    LabelMapper mapper;
    std::cout << "[Step 1] Label Mapping:\n";
    for (const auto& [name, label] : mapper.mappings()) {
        std::cout << "  Class: " << std::setw(5) << std::left << name
                  << " -> Label: " << label << "\n";
    }
    std::cout << "\n";

    // 3. Configure PreprocessingRunner
    PreprocessingConfig config;
    config.targetWidth          = 224;
    config.targetHeight         = 224;
    config.desiredChannels      = 3;
    config.padValue             = 0;
    config.maxSamplesPerClass   = maxSamplesPerClass;
    config.numThreads           = 0; // use all hardware threads for high performance

    PreprocessingRunner runner(config);
    PreprocessingSummary summary;

    // 4. Execute Full Incremental Pipeline
    std::cout << "[Step 2] Executing Preprocessing Pipeline...\n";
    bool success = runner.processDataset(datasetRoot, processedDir, summary);

    if (!success) {
        std::cerr << "[ERROR] Preprocessing failed!\n";
        return 1;
    }

    // 5. Print Detailed Summary
    PreprocessingRunner::printSummary(summary);

    // 6. Verification of Final Outputs
    std::cout << "[Step 3] Post-Processing Verification:\n";
    fs::path imagesDir = fs::path(processedDir) / "images";
    fs::path csvPath   = fs::path(processedDir) / "labels.csv";

    // Read labels.csv
    std::ifstream csv(csvPath.string());
    std::string line;
    std::getline(csv, line); // header

    size_t csvEntries = 0;
    size_t missingFiles = 0;
    size_t invalidDims = 0;
    uintmax_t totalPngBytes = 0;

    std::vector<std::string> sampleFilenames;

    ImageReader verifierReader(3);

    while (std::getline(csv, line)) {
        if (line.empty()) continue;
        auto comma = line.find(',');
        if (comma == std::string::npos) continue;

        std::string filename = line.substr(0, comma);
        while (!filename.empty() && std::isspace(static_cast<unsigned char>(filename.back()))) {
            filename.pop_back();
        }

        csvEntries++;

        fs::path imgPath = imagesDir / filename;
        if (!fs::exists(imgPath)) {
            missingFiles++;
        } else {
            totalPngBytes += fs::file_size(imgPath);
            int w = 0, h = 0, c = 0;
            if (!verifierReader.getInfo(imgPath.string(), w, h, c) || w != 224 || h != 224 || c != 3) {
                invalidDims++;
            }
            if (sampleFilenames.size() < 4) {
                sampleFilenames.push_back(imgPath.string());
            }
        }
    }

    std::cout << "  Total CSV records         : " << csvEntries << "\n";
    std::cout << "  Missing output files      : " << missingFiles << "\n";
    std::cout << "  Dimension failures        : " << invalidDims << "\n";
    std::cout << "  Total PNG storage size    : " << std::fixed << std::setprecision(2)
              << (totalPngBytes / (1024.0 * 1024.0)) << " MB (" << totalPngBytes << " bytes)\n";

    // 7. Verify Training-Time Conversion on a Sample Image
    std::cout << "\n[Step 4] Training-Time Representation Verification (float32 [0,1]):\n";
    if (!sampleFilenames.empty()) {
        std::string err;
        ImageData u8Img = verifierReader.read(sampleFilenames[0], err);
        NormalizedImage floatImg = ImageReader::toNormalizedFloat(u8Img);

        float minVal, maxVal;
        Normalization::computeMinMax(floatImg, minVal, maxVal);

        std::cout << "  Sample image              : " << sampleFilenames[0] << "\n";
        std::cout << "  Decoded uint8 shape       : " << u8Img.width << " x " << u8Img.height << " x " << u8Img.channels << "\n";
        std::cout << "  Converted float32 shape   : " << floatImg.width << " x " << floatImg.height << " x " << floatImg.channels << "\n";
        std::cout << "  Memory size in RAM        : " << floatImg.sizeBytes() << " bytes (exactly 602,112 bytes)\n";
        std::cout << "  Float pixel value range   : [" << std::fixed << std::setprecision(5) << minVal
                  << ", " << maxVal << "]\n";
        std::cout << "  Validation Status         : " << (Normalization::validate(floatImg) ? "PASSED (Strictly in [0, 1])" : "FAILED") << "\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << " Complete Dataset Preprocessing Execution FINISHED\n";
    std::cout << "========================================================\n";

    return 0;
}
