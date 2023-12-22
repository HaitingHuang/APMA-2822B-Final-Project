#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <cmath>
#include <sstream>
#include <cassert>
#include <set>
#include <cmath>
#include <cstdlib>

void calculateDistancesHost(const float *data, const float *center, float *distance_result, int N, int dimension, int k) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < k; ++col) {
            double sum = 0.0;  // Use double for higher precision
            for (int i = 0; i < dimension; ++i) {
                double diff = static_cast<double>(data[row * dimension + i]) - static_cast<double>(center[col * dimension + i]);
                sum += diff * diff;
            }
            distance_result[row * k + col] = static_cast<float>(sqrt(sum));  // Calculate the square root for Euclidean distance
        }
    }
}


// Function to initialize centroids using k-means++ method without parallelization
void initializeCentroidsKMeansPlusPlus(float *data, int n, int dataDimension, int k, float *center) {
    // std::random_device rd;
    std::mt19937 eng(42);
    std::uniform_int_distribution<> distr(0, n - 1);
    int firstCentroidIndex = distr(eng);
    for (int j = 0; j < dataDimension; ++j) {
        center[0 * dataDimension + j] = data[firstCentroidIndex * dataDimension + j];
    }

    std::vector<float> minDistances(n, std::numeric_limits<float>::max());

    for (int i = 1; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            float minDist = std::numeric_limits<float>::max();
            for (int c = 0; c < i; ++c) {
                float dist = 0.0;
                for (int dim = 0; dim < dataDimension; ++dim) {
                    float diff = data[j * dataDimension + dim] - center[c * dataDimension + dim];
                    dist += diff * diff;
                }
                minDist = std::min(minDist, dist);
            }
            minDistances[j] = minDist;
        }

        float totalMinDist = 0.0;
        for (int j = 0; j < n; ++j) {
            totalMinDist += minDistances[j];
        }

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

        for (int j = 0; j < dataDimension; ++j) {
            center[i * dataDimension + j] = data[nextCentroidIndex * dataDimension + j];
        }
    }
}

int main() {
    srand(42);
    auto start = std::chrono::high_resolution_clock::now();

    std::string filePath = "data/dataset_1000x10000.csv";
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Unable to open file\n";
        return 1;
    }

    std::string header;
    if (!std::getline(file, header)) {
        std::cerr << "Unable to read the header\n";
        return 1;
    }

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

    int n = allData.size();
    int dataDimension = 10000;
    int k = 2;

    float *data = new float[n * dataDimension];
    float *center = new float[k * dataDimension];

    int dataIndex = 0;
    for (const auto& point : allData) {
        for (float value : point) {
            data[dataIndex++] = value;
        }
    }

    initializeCentroidsKMeansPlusPlus(data, n, dataDimension, k, center);

    float *distance_result = new float[n * k];
    int *belongClass = new int[n];

    int iteration = 0;
    bool converged = false;
    float *centerNext = new float[k * dataDimension];
    for (int j = 0; j < dataDimension; ++j) {
        centerNext[j] = center[j];
    }

    float *centroidSums = new float[k * dataDimension]();
    int *counts = new int[k]();

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
        calculateDistancesHost(data, center, distance_result, n, dataDimension, k);

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

        for (int i = 0; i < k; ++i) {
            for (int dim = 0; dim < dataDimension; ++dim) {
                if (counts[i] > 0) {
                    centerNext[i * dataDimension + dim] = centroidSums[i * dataDimension + dim] / counts[i];
                }
            }
        }

        converged = true;
        for (int i = 0; i < k * dataDimension; ++i) {
            if (std::abs(center[i] - centerNext[i]) > 0.0001) {
                converged = false;
            }
            center[i] = centerNext[i];
        }

        // std::cout << "Iteration " << iteration << " Centroids:" << std::endl;
        // for (int i = 0; i < k; ++i) {
        //     std::cout << "Centroid " << i << ": (";
        //     for (int j = 0; j < dataDimension; ++j) {
        //         std::cout << center[i * dataDimension + j];
        //         if (j < dataDimension - 1) std::cout << ", ";
        //     }
        //     std::cout << ")" << std::endl;
        // }

        iteration++;
    }

    if (converged) {
        std::cout << "Converged after " << iteration << " iterations." << std::endl;
    } else {
        std::cout << "Reached maximum iterations." << std::endl;
    }

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

    delete[] centerNext;
    delete[] belongClass;
    delete[] distance_result;
    delete[] center;
    delete[] data;
    delete[] centroidSums;
    delete[] counts;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cerr << "Ending timer, time: " << duration.count() << " ms\n";

    return 0;
}
