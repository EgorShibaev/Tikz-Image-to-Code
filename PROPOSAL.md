### Project Proposal: Image-to-Code Translation for TikZ Diagrams

#### Project Title:
Deep Learning-Based Image-to-Code Translation for TikZ Diagrams

#### Participants:
- Egor Shibaev
- Alexander Kovrigin

#### Project Description:
The proposed project aims to develop a deep learning model capable of converting images of TikZ diagrams, including hand-drawn ones, into TikZ code. This model will facilitate the automation of diagram coding, reducing manual coding effort and enhancing productivity.

#### Proposed Architecture:
The architecture for this project is inspired by multi-modal Large Language Models (LLMs). The approach involves using a Vision Transformer (ViT) to process the input diagram image and generate a corresponding embedding. This embedding is then transformed via a linear layer to match the embedding space of subsequent tokens in a sequence model. The transformed embedding serves as the initial token in a sequence generation model, which outputs the corresponding TikZ code.

#### Dataset:
We plan to utilize the dataset available on Hugging Face at [HuggingFaceM4/datikz](https://huggingface.co/datasets/HuggingFaceM4/datikz). This dataset comprises 48,296 entries, each containing an image of a TikZ diagram and its corresponding code, providing a rich source for training and validating our model.

#### Previous Work:
Existing work in this domain includes AutomaTikZ, available at [AutomaTikZ GitHub](https://github.com/potamides/AutomaTikZ). This project developed a model that takes a textual description of a TikZ diagram and outputs the corresponding TikZ code. Our project extends this concept by translating visual inputs directly into code.

#### Possible Challenges and Solutions:
1. **Complexity of Diagrams:**
   - **Challenge:** The model may struggle with complex diagram concepts initially.
   - **Solution:** Start training with simpler diagrams to ensure the model can learn fundamental structures before introducing more complex diagrams.

2. **Input Variability:**
   - **Challenge:** The model may initially only handle perfectly rendered diagrams and not hand-drawn sketches.
   - **Solution:** Implement data augmentation techniques to train the model with a variety of diagram styles, including imperfections typical of hand-drawn images.
