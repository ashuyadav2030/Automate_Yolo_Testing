# 🧪 YOLOv8 Model Evaluation & Reporting Tool

This repository provides a Python-based pipeline to **evaluate a YOLOv8 object detection model** by comparing its predictions with ground truth labels and generating an Excel report.

---

## 📌 Overview

The pipeline performs the following tasks:

1. Loads a folder of test **images** and their corresponding **YOLO-format labels**.
2. Sends each image to a specified **API endpoint** where the model is hosted.
3. Receives detection **predictions** from the model (bounding boxes, classes, confidence scores).
4. Compares model predictions with the ground truth to:
   - Count objects
   - Match categories
   - Detect differences
5. Generates a **comprehensive Excel report** with:
   - Per-image evaluation
   - Brand/category-wise share
   - Accuracy summary

---

