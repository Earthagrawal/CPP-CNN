#include "LabelMapper.hpp"
#include <algorithm>
#include <cctype>

LabelMapper::LabelMapper() {
    // Default cat/dog mapping: Cat = 0, Dog = 1
    addMapping("Cat", 0);
    addMapping("Dog", 1);
}

std::string LabelMapper::toLower(const std::string& s) {
    std::string res = s;
    std::transform(res.begin(), res.end(), res.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return res;
}

void LabelMapper::addMapping(const std::string& className, int label) {
    classToLabel_[className] = label;
    labelToClass_[label]     = className;
}

int LabelMapper::getLabel(const std::string& className) const {
    auto it = classToLabel_.find(className);
    if (it != classToLabel_.end()) {
        return it->second;
    }
    // Case-insensitive fallback
    std::string target = toLower(className);
    for (const auto& pair : classToLabel_) {
        if (toLower(pair.first) == target) {
            return pair.second;
        }
    }
    return -1;
}

std::string LabelMapper::getClassName(int label) const {
    auto it = labelToClass_.find(label);
    return (it != labelToClass_.end()) ? it->second : "";
}

std::string LabelMapper::classFromPath(const std::string& filepath) const {
    // Works with both '/' and '\\' separators.
    std::string path = filepath;
    std::replace(path.begin(), path.end(), '\\', '/');

    if (!path.empty() && path.back() == '/') {
        path.pop_back();
    }

    auto lastSlash = path.rfind('/');
    if (lastSlash == std::string::npos) {
        return "";
    }

    auto secondLastSlash = path.rfind('/', lastSlash - 1);
    if (secondLastSlash == std::string::npos) {
        secondLastSlash = 0;
    } else {
        secondLastSlash += 1; // skip delimiter
    }

    return path.substr(secondLastSlash, lastSlash - secondLastSlash);
}

int LabelMapper::labelFromPath(const std::string& filepath) const {
    return getLabel(classFromPath(filepath));
}

size_t LabelMapper::numClasses() const {
    return classToLabel_.size();
}

const std::unordered_map<std::string, int>& LabelMapper::mappings() const {
    return classToLabel_;
}
