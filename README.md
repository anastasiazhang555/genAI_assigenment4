# GenAI Assignment 4 — CNN, GAN, EBM, Diffusion API

This project implements a unified FastAPI service that exposes training and generation endpoints for four models:

- **CNN Classifier** (trained on CIFAR-10)
- **GAN Generator** (trained on MNIST)
- **Energy-Based Model (EBM)** (trained on CIFAR-10)
- **Diffusion Model** (trained on CIFAR-10)

This API is based on the helper library developed in Assignment 2 & Assignment 3 and extended in Assignment 4 to include **EBM** and **Diffusion** generative models.

The code is fully modularized under the `helper_lib/` package and supports both training and sampling via HTTP routes.

---

## 📁 Project Structure
.
├── helper_lib/
│   ├── init.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── generator.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
│
├── main.py                # FastAPI server
├── requirements.txt       # Python dependencies
├── Dockerfile             # Optional containerization
└── .gitignore             # Ignore checkpoints & large files

---

## 🚀 How to Run the API

### 1. Install Dependencies

```bash
pip install -r requirements.txt

### 2. Start FastAPI Server
```bash
uvicorn main:app --reload

### 3. Open API Documentation (Swagger UI)
```bash
Once the server starts, open:

👉 http://127.0.0.1:8000/docs

You will see all available routes and can run training / generation directly in the browser.
