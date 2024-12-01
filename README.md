# TikZ Image2Code Model  

This repository provides tools for converting TikZ-generated images into LaTeX code using fine-tuned machine learning models.
Below are links to datasets, models, live demos, and other resources to help you use and understand the project.  

## Google Slides  
A detailed presentation explaining the project workflow, models, datasets, and results can be found [here](https://docs.google.com/presentation/d/1anlfVz0DYdcGS_io1uMEZkMAxXEOEOXKNoGQQUr9o0M/edit?usp=sharing).

---

## Project Resources  

### Dataset  
- **TikZ Short Code Dataset**: [EgorShibaev/TikZ-short-code](https://huggingface.co/datasets/EgorShibaev/TikZ-short-code)  
  A dataset of TikZ code and corresponding images. It was generated using GPT-3.5 and LaTeX tools to produce smaller, more manageable code examples for training models.

### Models  
1. **Pix2TikZ**:  
   - Repository: [pix2tikz Checkpoint](https://github.com/lukas-blecher/LaTeX-OCR)  
   - A fine-tuned Transformer model initially trained for parsing LaTeX formulas, now adapted for TikZ images.

2. **LLaVA Model**:  
   - Repository: [waleko/TikZ-llava-1.5-7b](https://huggingface.co/waleko/TikZ-llava-1.5-7b)  
   - A multimodal large language model fine-tuned for image-to-code tasks, specifically TikZ.

### Live Demo  
- **TikZ Assistant**: [Try the model](https://huggingface.co/spaces/waleko/TikZ-Assistant)  
  A web interface to test the model on TikZ images and receive corresponding LaTeX code.

---

## Code Repositories  
- **TikZ Image2Code**: [Repository](https://github.com/EgorShibaev/Tikz-Image-to-Code)  
  Contains scripts for training, fine-tuning, and deploying the models.

---

## Training Details  

### Pix2TikZ  
- **Optimizer**: Adam (learning rate: 0.001)  
- **Scheduler**: StepLR  
- **Batch Size**: 12  
- **Hardware**: Single V100S GPU (7 hours of training)  
- **Epochs**: 400  

### LLaVA  
- **Optimizer**: AdamW (initial learning rate: 5e-4)  
- **LoRA Configuration**: Rank=64, Alpha=6  
- **Batch Size**: 8  
- **Hardware**: Single A100 GPU (3 hours of training)  

---

## Future Work  
- Create an evaluation pipeline for model performance.  
- Expand the dataset to include more samples.  
- Use data augmentation techniques to support hand-drawn TikZ diagrams.
