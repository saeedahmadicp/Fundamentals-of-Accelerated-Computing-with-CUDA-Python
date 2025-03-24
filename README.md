# Fundamentals of Accelerated Computing with CUDA Python

This repository contains materials and exercises for the "Fundamentals of Accelerated Computing with CUDA Python" course.

## Introduction

This course provides hands-on experience accelerating Python applications using NVIDIA CUDA. You'll learn how to leverage NVIDIA GPUs to significantly improve the performance of your Python code through parallel computing techniques.

## Prerequisites

- Basic Python programming knowledge
- Familiarity with NumPy
- Understanding of basic parallel programming concepts (helpful but not required)
- NVIDIA GPU with CUDA support
- CUDA toolkit installed

## Repository Structure

This repository contains:
- Jupyter notebooks with guided exercises
- Sample solutions
- Supporting Python modules and utility files
- Data files used in exercises

## Setup Instructions

1. Install the required dependencies:
   ```
   pip install numba cuda-python cupy
   ```

2. Verify your CUDA installation:
   ```
   python -c "import numba; print(numba.cuda.is_available())"
   ```

3. Launch Jupyter notebooks:
   ```
   jupyter notebook
   ```

## How to Use

Work through the notebooks in sequential order. Each notebook introduces new concepts and builds upon previous lessons. Complete the exercises in each notebook to reinforce your learning.

## Course Completion

Upon completing all exercises in this repository, you will have gained practical experience in:
- Profiling Python applications to identify bottlenecks
- Using Numba to accelerate Python functions with CUDA
- Working with CUDA memory management
- Implementing parallel algorithms on the GPU
- Optimizing CUDA Python code for maximum performance

## Attribution

These materials are based on the NVIDIA Deep Learning Institute (DLI) course: "Fundamentals of Accelerated Computing with CUDA Python". 

Original course can be found at: [NVIDIA DLI - Fundamentals of Accelerated Computing with CUDA Python](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-AC-02+V1)

Copyright © NVIDIA Corporation. All rights reserved.