# ⚡ Snapdragon® Sentinel AI

> **100% On-Device, Air-Gapped Semantic Document Search powered by Qualcomm® Snapdragon® Hexagon™ NPU and DirectML.**

[![Qualcomm Snapdragon](https://img.shields.io/badge/Qualcomm-Snapdragon%20X%20Elite-red?style=for-the-badge&logo=qualcomm)](https://www.qualcomm.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-DirectML-blue?style=for-the-badge)](https://onnxruntime.ai/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## 📌 Executive Summary

**Snapdragon® Sentinel AI** is an enterprise-grade, edge-native document retrieval engine engineered specifically for **Qualcomm® Snapdragon® X Elite** and Snapdragon series processors. 

By offloading heavy vector embedding generation and tensor processing onto the **Qualcomm Hexagon™ NPU via DirectML**, Sentinel AI enables instant, air-gapped semantic search across local documents—eliminating reliance on cloud LLMs, external APIs, and active internet connectivity with **0 KB network payload**.

---

## 🚀 Key Features

* **⚡ Hardware Acceleration**: Executes ONNX embedding models directly on the Qualcomm Hexagon NPU using ONNX Runtime’s `DirectMLExecutionProvider`.
* **🔒 100% Air-Gapped Privacy**: Performs PDF ingestion, text cleaning, chunking, and vector scoring locally with zero cloud telemetry or data leakage.
* **📊 Live Latency Telemetry**: Real-time side-by-side performance analytics comparing Hexagon NPU acceleration against standard CPU execution.
* **🔍 Context Match & Extraction**: Ranks top relevant document chunks using cosine similarity and highlights matching query terms dynamically.
* **🛠️️ ONNX Session Inspector**: Low-level diagnostic viewer for ONNX tensor binding and precision layers (FP16/INT8).

---

## 📊 Performance Benchmark

| Execution Provider | Hardware Engine | Average Latency (ms) | Speedup Factor | Network Payload |
| :--- | :--- | :--- | :--- | :--- |
| **DirectMLExecutionProvider** | **Qualcomm Hexagon™ NPU** | **~780 ms** | **3.4x Faster 🚀** | **0 KB (Offline)** |
| CPUExecutionProvider | System CPU Baseline | ~2650 ms | 1.0x (Reference) | 0 KB (Offline) |

*Benchmarked on ONNX Runtime targeting DirectML DirectX 12 compute abstraction.*

---

## 🏗️ Architecture Overview
