#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <random>
#include <cmath>
#include <sstream>
#include <cassert>
#include <set>
#include <cstdlib>
#include <numeric>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Function declarations
__global__ void calculateDistancesBatched(const float* data, const float* center, float* distance_result, int N, int dimension, int k, int batchSize, int batchOffset) {

    //计算batch的索引
    int batchIndex = batchOffset + blockIdx.x * blockDim.x + threadIdx.x;
    int start = batchIndex * batchSize;
    int end = (start + batchSize < N) ? (start + batchSize) : N;

    for (int row = start; row < end; ++row) {
        for (int col = 0; col < k; ++col) {
            double sum = 0.0;
            for (int i = 0; i < dimension; ++i) {
                double diff = static_cast<double>(data[row * dimension + i]) - static_cast<double>(center[col * dimension + i]);
                sum += diff * diff;
            }
            distance_result[row * k + col] = sqrt(sum);
        }
    }
}

// Function to initialize centroids using k-means++ method with OpenMP
void initializeCentroidsKMeansPlusPlus(float* data, int n, int dataDimension, int k, float* center) {
    std::mt19937 eng(42);
    std::uniform_int_distribution<> distr(0, n - 1);

    // Randomly select the first centroid
    int firstCentroidIndex = distr(eng);
    for (int j = 0; j < dataDimension; ++j) {
        center[j] = data[firstCentroidIndex * dataDimension + j];
    }

    // Initialize a vector to store the squared distances to the closest centroid for each point
    std::vector<float> minDistances(n, std::numeric_limits<float>::max());

    // Iterate to select the remaining centroids
    for (int i = 1; i < k; ++i) {
        #pragma omp parallel for
        for (int j = 0; j < n; ++j) {
            float minDist = minDistances[j];
            for (int c = 0; c < i; ++c) {
                float dist = 0.0;
                for (int dim = 0; dim < dataDimension; ++dim) {
                    float diff = data[j * dataDimension + dim] - center[c * dataDimension + dim];
                    dist += diff * diff;
                }
                if (dist < minDist) {
                    minDist = dist;
                    break;
                }
            }
            minDistances[j] = minDist;
        }

        float totalMinDist = std::accumulate(minDistances.begin(), minDistances.end(), 0.0f);
        float randValue = std::uniform_real_distribution<float>(0, totalMinDist)(eng);
        float partialSum = 0.0;
        int nextCentroidIndex = -1;
        for (int j = 0; j < n; ++j) {
            partialSum += minDistances[j];
            if (partialSum >= randValue) {
                nextCentroidIndex = j;
                break;
            }
        }

        // Set the next centroid
        for (int j = 0; j < dataDimension; ++j) {
            center[i * dataDimension + j] = data[nextCentroidIndex * dataDimension + j];
        }
    }
}


int main() {
    srand(42);
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed;

    //start timer
    start = std::chrono::high_resolution_clock::now();
    std::cerr << "Loading data...\n";
    std::string filePath = "data/dataset_1000000x10.csv";
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Unable to open file\n";
        return 1;
    }


    // Skip the header
    std::string header;
    if (!std::getline(file, header)) {
        std::cerr << "Unable to read the header\n";
        return 1;
    }

    // Read the entire file in the main thread
    std::vector<std::vector<float>> allData;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<float> dataPoint;
        float value;

        while (ss >> value) {
            dataPoint.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        if (!dataPoint.empty()) {
            allData.push_back(dataPoint);
        }
    }
    file.close();
    std::cerr << "Data loaded...\n";

    int n = allData.size();
    int dataDimension = allData.empty() ? 0 : allData.front().size();
    std::cerr << "dataDimension: " << dataDimension << "\n";
    int k = 2; // Number of clusters

    // Flatten the data into a single array for CUDA
    float* data = new float[n * dataDimension];
    float* center = new float[k * dataDimension];

    int dataIndex = 0;
    for (const auto& point : allData) {
        for (float value : point) {
            data[dataIndex++] = value;
        }
    }

    std::cerr << "Initializing Centroids...\n";
    // start = std::chrono::high_resolution_clock::now();
    initializeCentroidsKMeansPlusPlus(data, n, dataDimension, k, center);
    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    std::cerr << "Centroids Initialized.\n";
    // std::cout << "Centroid Initialization Time: " << elapsed.count() << " seconds" << std::endl;

    float* distance_result = new float[n * k];
    int* belongClass = new int[n];

    int iteration = 0;
    bool converged = false;
    float* centerNext = new float[k * dataDimension];
    for (int j = 0; j < dataDimension; ++j) {
        centerNext[j] = center[j];
    }


    float* centroidSums = new float[k * dataDimension]();
    int* counts = new int[k]();

    const int numStreams = 8;
    cudaStream_t streams[numStreams];
    int batchSize;
    if (8 * numStreams > n) {
        batchSize = n;
    }
    else {
        batchSize = n / (8 * numStreams);
    }
    std::cerr << "Batch Size: " << batchSize << "\n";

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // std::chrono::duration<double> distance_time;
    // std::chrono::duration<double> centroid_time;

    float* d_data, * d_center, * d_distance_result;
    cudaMalloc(&d_data, n * dataDimension * sizeof(float));
    cudaMalloc(&d_center, dataDimension * k * sizeof(float));
    cudaMalloc(&d_distance_result, n * k * sizeof(float));
    //因为data不会变所以只需要拷贝一次
    cudaMemcpy(d_data, data, n * dataDimension * sizeof(float), cudaMemcpyHostToDevice);

    //计算每一个流需要处理的batch的个数
    int batchSizePerStream = ((n + batchSize - 1) / batchSize + numStreams - 1) / numStreams;
    //计算每一个流的线程数目
    int blockSize = batchSizePerStream < 1024 ? batchSizePerStream : 1024;
    int gridSize = batchSizePerStream < 1024 ? 1 : (batchSizePerStream + 1023) / 1024;

    while (iteration < 1000 && !converged) {
        // if (iteration == 0) {
        //     std::cout << "Initial Centroids:" << std::endl;
        //     for (int i = 0; i < k; ++i) {
        //         std::cout << "Centroid " << i << ": (";
        //         for (int j = 0; j < dataDimension; ++j) {
        //             std::cout << center[i * dataDimension + j];
        //             if (j < dataDimension - 1) std::cout << ", ";
        //         }
        //         std::cout << ")" << std::endl;
        //     }
        // }
        // start = std::chrono::high_resolution_clock::now();
        //calculateDistancesHost(data, center, distance_result, n, dataDimension, k, batchSize, streams, numStreams);

        {
            //center每次迭代都会变所以每次都需要拷贝到GPU
            cudaMemcpy(d_center, center, dataDimension* k * sizeof(float), cudaMemcpyHostToDevice);

            // std::cout << "Iteration " << iteration << " Centroids:" << std::endl;
            // for (int i = 0; i < k; ++i) {
            //     std::cout << "center " << i << ": (";
            //     for (int j = 0; j < dataDimension; ++j) {
            //         std::cout << center[i * dataDimension + j];
            //         if (j < dataDimension - 1) std::cout << ", ";
            //     }
            //     std::cout << ")" << std::endl;
            // }

            for (int i = 0; i < numStreams; ++i) {
                int batchOffset = batchSizePerStream * i;
                calculateDistancesBatched <<<gridSize, blockSize, 0, streams[i] >>> (d_data, d_center, d_distance_result, n, dataDimension, k, batchSize, batchOffset);
            }
            for (int i = 0; i < numStreams; ++i) {
                cudaStreamSynchronize(streams[i]);
            }
            //执行完毕后将距离结果拷贝到CPU
            cudaMemcpy(distance_result, d_distance_result, n * k * sizeof(float), cudaMemcpyDeviceToHost);
        }

        // end = std::chrono::high_resolution_clock::now();
        // elapsed = end - start;
        // distance_time += elapsed;
        // std::cout << "Calculated Distances (Iteration " << iteration << "):" << std::endl;
        // for (int i = 0; i < n; ++i) {
        //     std::cout << "Data Point " << i << ": ";
        //     for (int j = 0; j < k; ++j) {
        //         std::cout << distance_result[i * k + j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // Parallelize using OpenMP
        // start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for reduction(+:centroidSums[:k*dataDimension], counts[:k])
        for (int i = 0; i < n; ++i) {
            int closestCentroid = 0;
            float minDistance = distance_result[i * k];
            for (int j = 1; j < k; ++j) {
                if (distance_result[i * k + j] < minDistance) {
                    minDistance = distance_result[i * k + j];
                    closestCentroid = j;
                }
            }
            belongClass[i] = closestCentroid;
            counts[closestCentroid]++;
            for (int dim = 0; dim < dataDimension; ++dim) {
                centroidSums[closestCentroid * dataDimension + dim] += data[i * dataDimension + dim];
            }
        }

        // Update centroids
        for (int i = 0; i < k; ++i) {
            for (int dim = 0; dim < dataDimension; ++dim) {
                if (counts[i] > 0) {
                    centerNext[i * dataDimension + dim] = centroidSums[i * dataDimension + dim] / counts[i];
                }
            }
        }

        // end = std::chrono::high_resolution_clock::now();
        // elapsed = end - start;
        // centroid_time += elapsed;

        // Calculate new centroids and check for convergence
        converged = true;
        for (int i = 0; i < k * dataDimension; ++i) {
            if (std::abs(center[i] - centerNext[i]) > 0.0001) {
                converged = false;
                break;
            }
            // center[i] = centerNext[i];  // Update center
        }

        // Print centroids for the current iteration
        // std::cout << "Iteration " << iteration << " Centroids:" << std::endl;
        // for (int i = 0; i < k; ++i) {
        //     std::cout << "Centroid " << i << ": (";
        //     for (int j = 0; j < dataDimension; ++j) {
        //         std::cout << center[i * dataDimension + j];
        //         if (j < dataDimension - 1) std::cout << ", ";
        //     }
        //     std::cout << ")" << std::endl;
        // }

        if (!converged) {
            for (int i = 0; i < k * dataDimension; ++i) {
                center[i] = centerNext[i];
            }
            // Reset sums and counts only if we are going to do another iteration
            std::fill(centroidSums, centroidSums + k * dataDimension, 0);
            std::fill(counts, counts + k, 0);
        }

        iteration++;
    }
    std::cout << "Total Iterations: " << iteration << std::endl;

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    if (converged) {
        std::cout << "Converged after " << iteration << " iterations." << std::endl;
    }
    else {
        std::cout << "Reached maximum iterations." << std::endl;
    }
    // std::cout << "Distance Computation Time: " << distance_time.count() << " seconds" << std::endl;
    // std::cout << "Centroid Update Time: " << centroid_time.count() << " seconds" << std::endl;

    // Writing final assignments to file
    // std::ofstream outputFile("final_assignments.txt");
    // if (!outputFile.is_open()) {
    //     std::cerr << "Failed to open output file.\n";
    //     delete[] centerNext;
    //     delete[] belongClass;
    //     delete[] distance_result;
    //     delete[] center;
    //     delete[] data;
    //     return 1;
    // }

    // for (int i = 0; i < n; ++i) {
    //     outputFile << belongClass[i] << std::endl;
    // }

    // outputFile.close();

    // Cleaning up
    delete[] centerNext;
    delete[] belongClass;
    delete[] distance_result;
    delete[] center;
    delete[] data;
    delete[] centroidSums;
    delete[] counts;

    end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Ending timer, time: " << duration.count() << " ms" << std::endl;

    return 0;
}
