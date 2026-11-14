#pragma once

#include "../data/Dataset.hpp"
#include <string>
#include <vector>

// Statistics collected during image inspection.
struct InspectionStats {
    int totalFiles      = 0;
    int readableFiles   = 0;
    int corruptedFiles  = 0;
    int minWidth        = 0;
    int maxWidth        = 0;
    int minHeight       = 0;
    int maxHeight       = 0;
    double avgWidth     = 0.0;
    double avgHeight    = 0.0;
    int minChannels     = 0;
    int maxChannels     = 0;
    std::vector<std::string> corruptedPaths;
};

// Inspects images to collect dimension statistics and detect corrupted files.
class ImageInspector {
public:
    ImageInspector() = default;

    // Inspect a list of file paths using fast header decoding (stbi_info).
    InspectionStats inspect(const std::vector<std::string>& filepaths) const;

    // Inspect all entries in a Dataset.
    InspectionStats inspect(const Dataset& dataset) const;

    // Format inspection report as a string.
    static std::string generateReport(const InspectionStats& stats);

    // Print inspection report to stdout.
    static void printReport(const InspectionStats& stats);
};
