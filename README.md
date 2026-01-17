# 📌 Project Title

> ### **End-to-End Medical AI Inference and Web Serving**



<br><br>
## 📖 Overview

With the rapid aging of the population, the demand for medical services continues to increase, leading to a growing shortage of medical professionals. <br> 
To address this challenge, AI-driven medical devices and intelligent healthcare systems are being actively developed to support clinical decision-making and improve efficiency.<br>
In this project, I developed a web-based application that leverages both computer vision and large language models (LLMs) to assist medical professionals in the diagnostic process.<br> 
The system aims to enhance diagnostic accuracy, reduce the workload of clinicians, and demonstrate how AI models can be effectively deployed and served in a real-world medical environment.

<br><br>
## 🧩 Data Preprocessing
- **Dataset**: 10,000 Pediatric Abdominal X-ray Composite Images (Pyloric Stenosis, Pneumoperitoneum, Air Fluid Level, Constipation, Normal - 5 Classes)
- **Method**: Applied on-the-fly data augmentation using Albumentations, including geometric transformations (shift, rotation), photometric variations (brightness/contrast), and elastic deformation to improve model robustness and generalization.

<br><br>
## 🤖 Model Training
- ### **Vision**
  - **Pediatric Abdominal X-ray Diagnosis**
    - Based on the U-Net original paper, I trained a model with BatchNorm and Padding added for training stability and segmentation performance.
    - The model was trained using a custom loss function that combines two components:  
      1. **Multi-class Dice Loss** – to evaluate how accurately the model identifies affected regions.  
      2. **Weighted Cross-Entropy Loss** – to ensure proper class discrimination, with higher weight applied to the underrepresented classes since the 'Normal' class dominates the dataset.        
      
      By combining these, the custom loss balances both segmentation accuracy and class-wise classification performance.
    - Optimizer: AdamW, Learning Rate: 1e-4 (with ReduceLROnPlateau scheduler)
    - Epochs: 300 (with early stopping)
    - **Train Multi-Class Dice Score:** 0.898 → 0.908  
    - **Test Multi-Class Dice Score:** 0.883 → 0.912
    - Future improvements will focus on code refactoring and incorporating insights from recent research papers to further enhance training accuracy and model performance.

<br><br>
- ### **Large Language Model**
  - **snuh/hari-q3-14b**
    - This model is a Korean medical language model developed by Seoul National University Hospital. It has demonstrated strong performance, achieving 84.14% accuracy on the Korean Medical Licensing Examination (KMLE). We use this model to assist with diagnostic inference in a prototype system, utilizing LangChain templates and vLLM for initial deployment.
    - Future updates will focus on optimizing model serving for production environments.
  - If high-quality datasets become available, future work may include fine-tuning the model to further improve performance.

<br><br>
## 🛠️ Tech Stack

| Category        | Tools / Frameworks                |
|----------------|-----------------------------------|
| OS              | Ubuntu 22.04                      |
| Language        | Python 3.10, HTML5, CSS3, JavaScript(ES6) |
| Framework       | PyTorch, FastAPI, React     |
| Environment     | Jupyter Notebook / VSCode             |
| Database        | Redis                               |
| Hardware        | NVIDIA RTX 4090                      |


<br><br>
## 📂 Project Structure

```bash
.           
├── Vision/               
│   ├── Origin_UNet.py              # Model Training
│   ├── XRaySegModules.py           # Image Segmentation Modules 
│   └── [...]                
├── backend/                     # Web interface
│   ├── Modules/
│   │   ├── LLMModules.py           # Large Language Model Serving Modules
│   │   ├── TypeVariable.py         # Type Hint Variables
│   │   └── VisionModules.py        # Vision Model Serving Modules
│   ├── main.py                     # Backend Main File
│   └── [...]          
├── frontend/
│   ├── src/
│   │   ├── Component/              # Web Components
│   │   ├── Data/                   # Web Images
│   │   └── App.js                  # Frontend Main File
│   └── [...]  
└── [...]               
```

<br><br>
## 🚀 Future Work
- Improve image segmentation performance for pediatric abdominal X-rays by incorporating insights from recent research and state-of-the-art methods.  
- Expand the number of vision models to support a wider range of diagnostic tasks and improve clinical applicability.  
- Apply quantization to the hari-q3 model to optimize it for the available server resources.  
- If high-quality and sufficient datasets are available, consider fine-tuning the model to further enhance diagnostic accuracy.  
- Evaluate the current vLLM implementation within LangChain for production readiness, and if necessary, deploy a separate vLLM server for robust serving in real-world environments.  
- Deploy the system on AWS to enable scalable and reliable service operation.

<br><br>
## 📚 References
1. Seoul National University Hospital. snuh/hari-q3-14b: Korean Medical Language Model.
2. Olaf Ronneberger, Philiop Fischer, & Thomas Brox. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) (Vol. 9351, Issue Cvd, pp. 234-241).
3. Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, & Leonardo Rundo. (2022). Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation. In Computerized Medical Imaging and Graphics (Vol. 95, Article 102026)
