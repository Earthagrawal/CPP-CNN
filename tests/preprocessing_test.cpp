// CatDogCNN - Preprocessing & Data Handling Unit & Integration Tests

#include "../data/ImageReader.hpp"
#include "../data/LabelMapper.hpp"
#include "../data/Dataset.hpp"
#include "../preprocessing/ImageInspector.hpp"
#include "../preprocessing/Resize.hpp"
#include "../preprocessing/Padder.hpp"
#include "../preprocessing/Normalization.hpp"
#include "../preprocessing/PreprocessingRunner.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstdint>

namespace fs = std::filesystem;

static int testsPassed = 0;
static int testsFailed = 0;

#define TEST_CHECK(condition, message) do { \
    if (condition) { \
        std::cout << "  [PASS] " << message << "\n"; \
        testsPassed++; \
    } else { \
        std::cerr << "  [FAIL] " << message << " (Line " << __LINE__ << ")\n"; \
        testsFailed++; \
    } \
} while(0)

static std::string resolvePath(const std::string& p1, const std::string& p2, const std::string& p3) {
    if (fs::exists(p1)) return p1;
    if (fs::exists(p2)) return p2;
    if (fs::exists(p3)) return p3;
    return p1;
}

void testLabelMapper() {
    std::cout << "\n--- 1. Testing LabelMapper ---\n";
    LabelMapper mapper;

    TEST_CHECK(mapper.getLabel("Cat") == 0, "Cat maps to 0");
    TEST_CHECK(mapper.getLabel("Dog") == 1, "Dog maps to 1");
    TEST_CHECK(mapper.getLabel("cat") == 0, "Case-insensitive: cat maps to 0");
    TEST_CHECK(mapper.getLabel("DOG") == 1, "Case-insensitive: DOG maps to 1");
    TEST_CHECK(mapper.getLabel("Horse") == -1, "Unknown class returns -1");

    TEST_CHECK(mapper.getClassName(0) == "Cat", "0 maps to Cat");
    TEST_CHECK(mapper.getClassName(1) == "Dog", "1 maps to Dog");
    TEST_CHECK(mapper.getClassName(99) == "", "Unknown label returns empty string");

    TEST_CHECK(mapper.classFromPath("dataset/PetImages/Cat/100.jpg") == "Cat", "classFromPath with forward slash");
    TEST_CHECK(mapper.classFromPath("dataset\\PetImages\\Dog\\200.jpg") == "Dog", "classFromPath with backslash");
    TEST_CHECK(mapper.labelFromPath("dataset/PetImages/Cat/100.jpg") == 0, "labelFromPath returns 0 for Cat");
    TEST_CHECK(mapper.labelFromPath("dataset/PetImages/Dog/200.jpg") == 1, "labelFromPath returns 1 for Dog");
}

void testDataset() {
    std::cout << "\n--- 2. Testing Dataset ---\n";
    Dataset dataset;
    dataset.addEntry("path/to/cat1.jpg", 0, "Cat");
    dataset.addEntry("path/to/cat2.jpg", 0, "Cat");
    dataset.addEntry("path/to/dog1.jpg", 1, "Dog");

    TEST_CHECK(dataset.size() == 3, "Dataset size is 3 after adding entries");
    TEST_CHECK(!dataset.empty(), "Dataset is not empty");
    TEST_CHECK(dataset[0].className == "Cat", "First entry is Cat");
    TEST_CHECK(dataset[2].className == "Dog", "Third entry is Dog");

    auto sub = dataset.subset(2);
    TEST_CHECK(sub.size() == 2, "subset(2) returns 2 elements");

    auto balanced = dataset.getBalancedSubset(1);
    TEST_CHECK(balanced.size() == 2, "getBalancedSubset(1) returns 2 elements (1 Cat, 1 Dog)");
    TEST_CHECK(balanced[0].label == 0 && balanced[1].label == 1, "Balanced subset has 1 of each class");

    TEST_CHECK(Dataset::isSupportedImage("test.jpg"), "Supports .jpg");
    TEST_CHECK(Dataset::isSupportedImage("test.JPEG"), "Supports .JPEG");
    TEST_CHECK(Dataset::isSupportedImage("test.png"), "Supports .png");
    TEST_CHECK(Dataset::isSupportedImage("test.bmp"), "Supports .bmp");
    TEST_CHECK(!Dataset::isSupportedImage("test.txt"), "Rejects .txt");
}

void testImageReader() {
    std::cout << "\n--- 3. Testing ImageReader & Training-Time Conversion ---\n";
    ImageReader reader(3); // RGB

    std::string err;
    ImageData bad = reader.read("non_existent_file_xyz.jpg", err);
    TEST_CHECK(bad.empty(), "Reading nonexistent file returns empty ImageData");
    TEST_CHECK(!err.empty(), "Error message populated on missing file");

    int w = 0, h = 0, c = 0;
    bool infoBad = reader.getInfo("non_existent_file_xyz.jpg", w, h, c);
    TEST_CHECK(!infoBad, "getInfo returns false on missing file");

    // Test uint8 to float32 training conversion
    ImageData testImg;
    testImg.width = 1;
    testImg.height = 1;
    testImg.channels = 3;
    testImg.pixels = {128, 64, 255}; // R, G, B

    NormalizedImage floatImg = ImageReader::toNormalizedFloat(testImg);
    TEST_CHECK(floatImg.width == 1 && floatImg.height == 1 && floatImg.channels == 3, "Shape preserved (1x1x3)");
    TEST_CHECK(std::abs(floatImg.pixels[0] - 0.50196f) < 1e-4f, "R: 128 / 255 = 0.50196");
    TEST_CHECK(std::abs(floatImg.pixels[1] - 0.25098f) < 1e-4f, "G: 64 / 255 = 0.25098");
    TEST_CHECK(std::abs(floatImg.pixels[2] - 1.00000f) < 1e-4f, "B: 255 / 255 = 1.00000");
}

void testResize() {
    std::cout << "\n--- 4. Testing Resize (Aspect Ratio Preservation) ---\n";
    Resize resizer(224, 224);

    // Wide image (400 x 200, aspect ratio 2:1)
    ImageData wide;
    wide.width = 400;
    wide.height = 200;
    wide.channels = 3;
    wide.pixels.resize(400 * 200 * 3, 100);

    ImageData resizedWide = resizer.apply(wide);
    TEST_CHECK(resizedWide.width == 224, "Wide image: width fills target 224");
    TEST_CHECK(resizedWide.height == 112, "Wide image: height scaled to 112 (preserving 2:1 aspect ratio)");
    TEST_CHECK(resizedWide.channels == 3, "Channels preserved");

    // Tall image (150 x 300, aspect ratio 1:2)
    ImageData tall;
    tall.width = 150;
    tall.height = 300;
    tall.channels = 3;
    tall.pixels.resize(150 * 300 * 3, 200);

    ImageData resizedTall = resizer.apply(tall);
    TEST_CHECK(resizedTall.width == 112, "Tall image: width scaled to 112 (preserving 1:2 aspect ratio)");
    TEST_CHECK(resizedTall.height == 224, "Tall image: height fills target 224");
}

void testPadder() {
    std::cout << "\n--- 5. Testing Padder (Centered Padding & Odd Remainders) ---\n";
    Padder padder(224, 224, 0);

    ImageData wide;
    wide.width = 224;
    wide.height = 112;
    wide.channels = 3;
    wide.pixels.resize(224 * 112 * 3, 255); // all white

    ImageData paddedWide = padder.apply(wide);
    TEST_CHECK(paddedWide.width == 224, "Padded wide: width is 224");
    TEST_CHECK(paddedWide.height == 224, "Padded wide: height is 224");
    TEST_CHECK(paddedWide.pixels[0] == 0, "Top padding border is 0 (black)");

    size_t centerIdx = (112 * 224 + 112) * 3;
    TEST_CHECK(paddedWide.pixels[centerIdx] == 255, "Center pixel is from original content");

    ImageData oddImg;
    oddImg.width = 113;
    oddImg.height = 224;
    oddImg.channels = 3;
    oddImg.pixels.resize(113 * 224 * 3, 128);

    ImageData paddedOdd = padder.apply(oddImg);
    TEST_CHECK(paddedOdd.width == 224, "Odd width padded to exactly 224");
    TEST_CHECK(paddedOdd.height == 224, "Odd height is exactly 224");
}

void testNormalization() {
    std::cout << "\n--- 6. Testing Normalization ---\n";

    ImageData img;
    img.width = 2;
    img.height = 2;
    img.channels = 1;
    img.pixels = {0, 128, 255, 64};

    NormalizedImage nImg = Normalization::apply(img);
    TEST_CHECK(nImg.width == 2 && nImg.height == 2 && nImg.channels == 1, "Dimensions preserved");
    TEST_CHECK(std::abs(nImg.pixels[0] - 0.0f) < 1e-6f, "Pixel 0 normalized to 0.0");
    TEST_CHECK(std::abs(nImg.pixels[2] - 1.0f) < 1e-6f, "Pixel 255 normalized to 1.0");

    TEST_CHECK(Normalization::validate(nImg), "validate() returns true for valid normalized image");

    NormalizedImage invalid = nImg;
    invalid.pixels[0] = 1.5f;
    TEST_CHECK(!Normalization::validate(invalid), "validate() catches values > 1.0");

    float outMin, outMax;
    Normalization::computeMinMax(nImg, outMin, outMax);
    TEST_CHECK(outMin >= 0.0f && outMax <= 1.0f, "computeMinMax confirms [0, 1] range");
}

void testEndToEndPipeline(const std::string& datasetRoot, const std::string& testOutputDir) {
    std::cout << "\n--- 7. Testing End-to-End Pipeline & PNG Storage ---\n";

    if (!fs::exists(datasetRoot)) {
        std::cout << "  [SKIP] Dataset root not found at " << datasetRoot << ", skipping live image tests.\n";
        return;
    }

    PreprocessingConfig config;
    config.targetWidth = 224;
    config.targetHeight = 224;
    config.desiredChannels = 3;
    config.padValue = 0;
    config.maxSamplesPerClass = 2; // 2 Cat + 2 Dog = 4 samples
    config.numThreads = 2;

    PreprocessingRunner runner(config);
    PreprocessingSummary summary;

    bool ok = runner.processDataset(datasetRoot, testOutputDir, summary);
    TEST_CHECK(ok, "processDataset succeeded");
    TEST_CHECK(summary.successCount == 4, "Successfully processed 4 samples (2 Cat, 2 Dog)");
    TEST_CHECK(summary.failedCount == 0, "No sample failures in clean subset");

    // Verify output PNG files on disk
    fs::path samplePng = fs::path(testOutputDir) / "images" / "cat_00001.png";
    TEST_CHECK(fs::exists(samplePng), "Output PNG file exists on disk");

    ImageReader reader(3);
    int w = 0, h = 0, c = 0;
    TEST_CHECK(reader.getInfo(samplePng.string(), w, h, c) && w == 224 && h == 224 && c == 3,
               "Output PNG has exact dimensions 224x224x3");

    // Verify labels.csv
    fs::path labelsCsv = fs::path(testOutputDir) / "labels.csv";
    TEST_CHECK(fs::exists(labelsCsv), "labels.csv exists on disk");

    std::ifstream csvIn(labelsCsv.string());
    std::string header;
    std::getline(csvIn, header);
    TEST_CHECK(header == "filename,label", "labels.csv has correct header (filename,label)");

    // Test safe resumption
    PreprocessingSummary resumeSummary;
    runner.processDataset(datasetRoot, testOutputDir, resumeSummary);
    TEST_CHECK(resumeSummary.skippedValidCount == 4, "Safe resumption: skipped 4 already valid outputs");
}

int main(int argc, char* argv[]) {
    std::cout << "========================================================\n";
    std::cout << "   CatDogCNN - Preprocessing Pipeline Test Suite        \n";
    std::cout << "========================================================\n";

    std::string datasetRoot = resolvePath("../../dataset/PetImages", "../dataset/PetImages", "dataset/PetImages");
    std::string testOutputDir = resolvePath("../../dataset/processed_test", "../dataset/processed_test", "dataset/processed_test");

    if (argc >= 2) datasetRoot = argv[1];
    if (argc >= 3) testOutputDir = argv[2];

    testLabelMapper();
    testDataset();
    testImageReader();
    testResize();
    testPadder();
    testNormalization();
    testEndToEndPipeline(datasetRoot, testOutputDir);

    std::cout << "\n========================================================\n";
    std::cout << " Test Summary: " << testsPassed << " PASSED, "
              << testsFailed << " FAILED\n";
    std::cout << "========================================================\n\n";

    // Clean up temporary test output dir
    try {
        if (fs::exists(testOutputDir)) {
            fs::remove_all(testOutputDir);
        }
    } catch (...) {}

    return (testsFailed == 0) ? 0 : 1;
}
