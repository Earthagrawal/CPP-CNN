#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

// Maps class directory names to integer labels.
// Default mapping: "Cat" -> 0, "Dog" -> 1.
class LabelMapper {
public:
    LabelMapper();

    // Add or override a class mapping.
    void addMapping(const std::string& className, int label);

    // Look up a label by class name. Returns -1 if unknown.
    int getLabel(const std::string& className) const;

    // Look up a class name by label. Returns "" if unknown.
    std::string getClassName(int label) const;

    // Infer the class name from a file path by examining the parent directory.
    // e.g. ".../PetImages/Cat/123.jpg" -> "Cat"
    std::string classFromPath(const std::string& filepath) const;

    // Infer label directly from file path. Returns -1 if unmapped.
    int labelFromPath(const std::string& filepath) const;

    // Number of registered classes.
    size_t numClasses() const;

    // All registered mappings.
    const std::unordered_map<std::string, int>& mappings() const;

private:
    std::unordered_map<std::string, int> classToLabel_;
    std::unordered_map<int, std::string> labelToClass_;

    static std::string toLower(const std::string& s);
};
