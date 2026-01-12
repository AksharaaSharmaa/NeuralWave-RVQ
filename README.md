<div align="center">

# <u>🎵 NeuralWave-RVQ: Neural Audio Codec</u>

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Audio](https://img.shields.io/badge/Domain-Audio%20Signal%20Processing-violet)
![License](https://img.shields.io/badge/License-MIT-green)

**A lightweight, educational implementation of state-of-the-art Neural Audio Compression techniques.**

</div>

---

## 📖 Overview

**NeuralWave-RVQ** is a PyTorch implementation of a neural audio codec. Unlike traditional codecs (MP3, AAC) that rely on fixed algorithms, this project uses deep learning to learn efficient representations of sound.

It implements the core architecture found in modern generative audio models (like **Meta's EnCodec** or **Google's SoundStream**), specifically focusing on **Residual Vector Quantization (RVQ)** to create discrete latent spaces suitable for downstream tasks like Audio Generation (MusicGen/AudioLM).

---

## 🚀 Key Features

* **🧠 Convolutional Encoder/Decoder:** Compresses 16kHz audio into a low-dimensional latent space (downsampling factor of 128).
* **🔢 Residual Vector Quantization (RVQ):** Uses hierarchical codebooks to discretize continuous vectors, effectively compressing the signal while retaining high fidelity.
* **📉 Multi-Scale STFT Loss:** Trains on frequency-domain errors across multiple resolutions to ensure the reconstruction sounds natural and crisp.
* **⚡ Straight-Through Estimator:** Implements gradient flow through non-differentiable quantization steps.

---

## 🛠️ Architecture

The pipeline consists of three main stages:

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐      ┌──────────────┐      ┌───────────────────┐
│ Input Audio │ ───> │ Conv Encoder │ ───> │ Residual VQ     │ ───> │ Conv Decoder │ ───> │ Reconstructed     │
│             │      │              │      │ (Multi-Codebook)│      │              │      │ Audio             │
└─────────────┘      └──────────────┘      └─────────────────┘      └──────────────┘      └───────────────────┘
```

1. **Encoder:** Extracts features and downsamples the waveform.
2. **Quantizer (RVQ):** Maps features to the nearest codebook vectors recursively (Coarse → Fine details).
3. **Decoder:** Upsamples the quantized vectors back to the raw waveform.

---

## 💻 Installation & Usage

### Prerequisites

* Python 3.8+
* PyTorch 2.0+
* Torchaudio
* Numpy/Scipy

### Quick Start

#### 1. Clone the repository

```bash
git clone https://github.com/AksharaaSharmaa/NeuralWave-RVQ.git
cd NeuralWave-RVQ
```

#### 2. Install dependencies

```bash
pip install torch torchaudio numpy scipy soundfile matplotlib
```

#### 3. Run the Training Notebook

Open `Simple_Neural_Audio_Codec.ipynb` in Jupyter or Google Colab to train the model on your own audio sample.

---

## 🔬 Mathematical Concepts

This project implements **Residual Vector Quantization**. Instead of using a single massive codebook (which is computationally expensive), we use a series of smaller codebooks (C₁, C₂, ... Cₙ).

The quantized representation is computed as:

```
X̂ = Σ(i=1 to N) Quantize(Rᵢ₋₁, Cᵢ)
```

Where **R** is the residual error. This allows the model to scale bitrate easily—using more codebooks yields higher quality, while fewer codebooks yield higher compression.

### Key Benefits:
- **Scalable bitrate control:** Adjust compression by changing number of codebooks
- **Hierarchical representation:** Coarse-to-fine audio details
- **Efficient training:** Smaller codebooks converge faster

---

## 📊 Training Details

The model is trained using:

- **Multi-Scale STFT Loss:** Captures frequency-domain reconstruction quality across multiple time-frequency resolutions
- **Commitment Loss:** Encourages encoder outputs to stay close to codebook vectors
- **Codebook Loss:** Updates codebook vectors to match encoder outputs
- **Straight-Through Estimator:** Enables gradient flow through discrete quantization

---

## 🎯 Use Cases

This codec can serve as a foundation for:

- **Audio compression** at various bitrates
- **Generative audio models** (similar to MusicGen, AudioLM)
- **Audio editing** in latent space
- **Multi-modal AI systems** requiring discrete audio tokens
- **Research** in neural signal processing

---

## 🤝 Acknowledgements

This project is an implementation based on concepts introduced in:

* **SoundStream: An End-to-End Neural Audio Codec** (Google Research)
* **High Fidelity Neural Audio Compression** (Meta AI / EnCodec)

---

## 👨‍💻 Author

**Akshara**

* Computer Science Engineering Student
* Research Interests: Multimodal AI, RAG, Audio Signal Processing

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🌟 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/NeuralWave-RVQ/issues).

---

<div align="center">

**If you find this project useful, please consider giving it a ⭐!**

</div>
