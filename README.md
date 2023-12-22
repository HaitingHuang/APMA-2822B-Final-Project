# K-Means Clustering Optimization with CUDA and OpenMP

# Introduction
This project implements the K-Means clustering algorithm, a popular unsupervised classification method. It divides data into clusters based on distance metrics, iterating through centroid initialization, cluster assignments, and centroid updates. We focus on K-Means++ for efficient centroid initialization, Euclidean distance for assignments, and average-based updates.

# Optimization Techniques
OpenMP: Utilizes multi-core CPU architectures for parallel processing during cluster assignments and centroid updates.
CUDA: Leverages GPU's parallel capabilities for efficient distance calculations in large datasets.

# Implementation
Developed in C++, this project employs sophisticated techniques like K-Means++ for initialization and roofline analysis for performance evaluation. We use the Nvidia NSight profiler for CUDA optimization insights.

# Performance Analysis
A comparison between this optimized implementation and a serial version of K-Means clustering demonstrates significant improvements in efficiency and speed.
