# 🖼️ Generative AI Images — Stable Diffusion Pipeline

Project ini merupakan implementasi pipeline **Generative AI untuk image generation** menggunakan model Stable Diffusion. Dikembangkan sebagai submission kelas **Belajar Fundamental Generative AI (BFGAI)**, project ini mencakup eksplorasi Text-to-Image, Image-to-Image, Inpainting, Outpainting, hingga deployment menggunakan Streamlit.

---

## 📌 Fitur Utama

- **Text-to-Image Generation** — Generate gambar dari teks prompt menggunakan Stable Diffusion v1.5
- **Hyperparameter Tuning** — Eksplorasi pengaruh `guidance_scale` dan `num_inference_steps` terhadap kualitas gambar
- **Batch Inference** — Generate beberapa gambar dengan variasi inference steps sekaligus
- **Scheduler Comparison** — Perbandingan hasil generate menggunakan Euler A, DPM++ 2M Karras, dan DDIM
- **Image-to-Image (SDXL)** — Refinement gambar menggunakan Stable Diffusion XL Base + Refiner
- **Inpainting** — Edit bagian tertentu dari gambar menggunakan mask
- **CLIPSeg Segmentation** — Segmentasi otomatis menggunakan CLIPSeg untuk pembuatan mask
- **Outpainting** — Memperluas canvas gambar secara generatif
- **Streamlit App** — UI interaktif untuk mencoba semua fitur di atas secara langsung

---

## 🗂️ Struktur Project

```
Gen_AI_Images/
├── Pipeline_submission_BFGAI_Muhammad_Rizki_Yanuar.ipynb   # Pipeline utama (eksplorasi & eksperimen)
├── Streamlit_submission_BFGAI_Muhammad_Rizki_Yanuar.ipynb  # Notebook untuk deploy Streamlit
└── requirements.txt                                         # Dependensi project
```
---

## ⚙️ Teknologi yang Digunakan

| Teknologi | Keterangan |
|---|---|
| Python | Bahasa pemrograman utama |
| PyTorch | Framework deep learning |
| 🤗 Diffusers | Library untuk Stable Diffusion pipeline |
| 🤗 Transformers | Library untuk CLIPSeg segmentation |
| Stable Diffusion v1.5 | Model Text-to-Image & Inpainting |
| Stable Diffusion XL | Model Image-to-Image (Base + Refiner) |
| CLIPSeg | Model segmentasi berbasis teks |
| Streamlit | Framework web app interaktif |
| PIL (Pillow) | Manipulasi gambar |
| pyngrok | Tunnel untuk expose Streamlit dari Colab |
| Google Colab | Environment eksekusi (GPU) |

---

## 🚀 Cara Menjalankan

### Prasyarat

- Akun Google Colab (disarankan menggunakan GPU runtime)
- Akun Hugging Face dengan akses ke model Stable Diffusion
- Token Hugging Face (simpan di Colab Secrets dengan key `Hugging`)

### 1. Pipeline Notebook

Buka `Pipeline_submission_BFGAI_Muhammad_Rizki_Yanuar.ipynb` di Google Colab, lalu jalankan cell secara berurutan:

```bash
# Install dependencies
!pip install diffusers transformers scipy ftfy accelerate safetensors
```

```python
# Login ke Hugging Face
from huggingface_hub import login
login(token="your_hf_token")
```

### 2. Streamlit App

Buka `Streamlit_submission_BFGAI_Muhammad_Rizki_Yanuar.ipynb` di Google Colab, lalu:

```bash
# Install dependencies
!pip install -q pyngrok streamlit torch diffusers transformers streamlit_drawable_canvas==0.8.0
```

Jalankan semua cell untuk meng-generate `logic.py` dan menjalankan Streamlit via ngrok.

---

## 📊 Eksperimen & Temuan

### Guidance Scale
| CFG | Hasil |
|---|---|
| Rendah (3) | Gambar kurang sesuai prompt, warna tidak akurat |
| Tinggi (8) | Gambar detail dan sesuai prompt |

### Inference Steps
| Steps | Hasil |
|---|---|
| Rendah (10) | Gambar blur dan belum selesai di-generate |
| Tinggi (50) | Gambar tajam dan detail |

### Scheduler
| Scheduler | Karakteristik |
|---|---|
| Euler A | Detail halus, tapi warna bisa tidak akurat |
| DPM++ 2M Karras | Cukup sesuai prompt, sedikit blur |
| DDIM | Tekstur detail, tapi anatomi bisa tidak konsisten |

---

## 🤖 Model yang Digunakan

- [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) — Text-to-Image
- [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting) — Inpainting
- [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) — Image-to-Image (SDXL)
- [`CIDAS/clipseg-rd64-refined`](https://huggingface.co/CIDAS/clipseg-rd64-refined) — Text-guided Segmentation

---

## 👤 Author

**Muhammad Rizki Yanuar**  
Submission — Belajar Fundamental Generative AI (BFGAI)
