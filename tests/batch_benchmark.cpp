#include "../core/CUDABackend.hpp"
#include "../model/CatDogCNNModel.hpp"
#include "../loss/BCEWithLogitsLoss.hpp"
#include "../optimizers/SGD.hpp"
#include "../data/DataLoader.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

void benchmarkBatchSize(int batchSize, int numBatches = 5) {
    std::cout << "\n========================================================\n";
    std::cout << " Testing Batch Size: " << batchSize << "\n";
    std::cout << "========================================================\n";

    try {
        std::string manifestCsv = "dataset/split_manifest.csv";
        std::string imagesDir = "dataset/processed/images";

        // Create DataLoader
        DataLoader loader(manifestCsv, imagesDir, "train", batchSize, true, DeviceType::CUDA);

        // Fresh model & optimizer
        CatDogCNNModel model(DeviceType::CUDA);
        model.train(true);
        BCEWithLogitsLoss criterion;
        SGD optimizer(model.getParameters(), 0.01f, 0.9f, 1e-4f);

#ifdef USE_CUDA
        size_t freeMem = 0, totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        double usedMB = static_cast<double>(totalMem - freeMem) / (1024.0 * 1024.0);
        double totalMB = static_cast<double>(totalMem) / (1024.0 * 1024.0);
        std::cout << " Initial VRAM Used: " << std::fixed << std::setprecision(1)
                  << usedMB << " MB / " << totalMB << " MB ("
                  << (usedMB / totalMB * 100.0) << "%)\n";
#endif

        // Warmup on 1 batch
        std::cout << " Running warmup batch...\n";
        {
            Batch warmupBatch = loader.nextBatch();
            model.zeroGrad();
            Tensor logits = model.forward(warmupBatch.images);
            float loss = criterion.forward(logits, warmupBatch.labels);
            Tensor gradLogits = criterion.backward();
            model.backward(gradLogits);
            optimizer.step();
            CUDABackend::synchronize();
        }

#ifdef USE_CUDA
        cudaMemGetInfo(&freeMem, &totalMem);
        usedMB = static_cast<double>(totalMem - freeMem) / (1024.0 * 1024.0);
        std::cout << " Post-Warmup VRAM Used: " << usedMB << " MB / " << totalMB << " MB ("
                  << (usedMB / totalMB * 100.0) << "%)\n";
#endif

        std::cout << " Benchmarking next " << numBatches << " batches...\n";

        double totalLoadTime = 0.0;
        double totalGpuTime = 0.0;
        double totalBatchTime = 0.0;

        for (int i = 0; i < numBatches && loader.hasNext(); ++i) {
            auto t0 = std::chrono::steady_clock::now();

            Batch batch = loader.nextBatch();

            auto t1 = std::chrono::steady_clock::now();

            model.zeroGrad();
            Tensor logits = model.forward(batch.images);
            float loss = criterion.forward(logits, batch.labels);
            Tensor gradLogits = criterion.backward();
            model.backward(gradLogits);
            optimizer.step();
            CUDABackend::synchronize();

            auto t2 = std::chrono::steady_clock::now();

            double loadMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double gpuMs  = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double totalMs = std::chrono::duration<double, std::milli>(t2 - t0).count();

            totalLoadTime += loadMs;
            totalGpuTime += gpuMs;
            totalBatchTime += totalMs;

            std::cout << "   [Batch " << (i + 1) << "/" << numBatches << "] "
                      << "Total: " << std::setw(6) << std::setprecision(1) << totalMs << " ms "
                      << "(Load: " << loadMs << " ms | GPU Fwd+Bwd+Opt: " << gpuMs << " ms) | "
                      << "Loss: " << std::setprecision(4) << loss << "\n";
        }

        double avgBatchMs = totalBatchTime / numBatches;
        double avgLoadMs  = totalLoadTime / numBatches;
        double avgGpuMs   = totalGpuTime / numBatches;
        int totalDatasetBatches = (19992 + batchSize - 1) / batchSize;
        double estEpochMin = (avgBatchMs * totalDatasetBatches) / (1000.0 * 60.0);

        std::cout << " --------------------------------------------------------\n";
        std::cout << " Average Time per Batch  : " << std::fixed << std::setprecision(1) << avgBatchMs << " ms ("
                  << (avgBatchMs / 1000.0) << " s)\n";
        std::cout << "   - Avg Disk+CPU Load   : " << avgLoadMs << " ms\n";
        std::cout << "   - Avg GPU Fwd+Bwd+Step: " << avgGpuMs << " ms\n";
        std::cout << " Total Batches in 1 Epoch: " << totalDatasetBatches << " batches\n";
        std::cout << " Estimated 1-Epoch Train : " << std::setprecision(1) << estEpochMin << " minutes\n";

    } catch (const std::exception& ex) {
        std::cerr << "[-] Error or OOM on Batch Size " << batchSize << ": " << ex.what() << "\n";
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "   CatDogCNN Batch Size vs Processing Time Benchmark    \n";
    std::cout << "========================================================\n";

    if (!CUDABackend::isAvailable()) {
        std::cerr << "[-] CUDA not available!\n";
        return 1;
    }

    CUDADeviceInfo dev = CUDABackend::getDeviceInfo(0);
    std::cout << " Active GPU: " << dev.name << " (" << dev.totalGlobalMem / (1024*1024) << " MB VRAM)\n";

    // Test 32, 64, 128
    benchmarkBatchSize(32, 5);
    benchmarkBatchSize(64, 5);
    benchmarkBatchSize(128, 5);

    std::cout << "\n========================================================\n";
    std::cout << " Benchmark completed successfully!\n";
    std::cout << "========================================================\n";
    return 0;
}
