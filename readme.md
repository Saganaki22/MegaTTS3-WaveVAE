<div align="center">
    <h1>
    MegaTTS3-WaveVAE <img src="./assets/fig/Hi.gif" width="40px">
    </h1>
    <p>
    Unofficial Windows-Compatible Implementation<br>
    </p>
</div>
<div align="center">
    <a href="https://github.com/Saganaki22/MegaTTS3-WaveVAE"><img src="https://img.shields.io/badge/GitHub-Repository-black?logo=github" alt="GitHub"></a>
    <a href="https://github.com/Saganaki22/MegaTTS3-WaveVAE/commits/main"><img src="https://img.shields.io/github/last-commit/Saganaki22/MegaTTS3-WaveVAE" alt="Last Commit"></a>
    <a href="#"><img src="https://img.shields.io/badge/Platform-Windows-blue?logo=windows" alt="Platform"></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.10-brightgreen?logo=python" alt="Python"></a>
    <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.3.0+-orange?logo=pytorch" alt="PyTorch"></a>
    <a href="https://github.com/Saganaki22/MegaTTS3-WaveVAE/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</div>
<div align="center">
    <img src="https://img.shields.io/badge/Original%20by-Bytedance-%230077B5.svg?&style=flat-square&logo=bytedance&logoColor=white" />
    <img src="https://img.shields.io/badge/Windows%20Fork-Saganaki22-%230077B5.svg?&style=flat-square&logo=github&logoColor=white" />
</div>

## About This Fork

This is an **unofficial Windows-compatible fork** of the original [ByteDance MegaTTS3](https://github.com/bytedance/MegaTTS3) repository. This version includes:

- ✅ **Windows Compatibility**: Pre-configured for Windows installation
- ✅ **GPU Support**: Optimized PyTorch installations for RTX 30xx/40xx and RTX 50xx series
- ✅ **Simplified Setup**: Streamlined installation process with proper dependencies
- ✅ **Enhanced Web UI**: Improved Gradio interface with examples and better user experience
- ✅ **WaveVAE Included**: Includes the WaveVAE encoder/decoder thanks to [ACoderPassBy/MegaTTS-SFT](https://modelscope.cn/models/ACoderPassBy/MegaTTS-SFT)

## Key Features
- 🚀**Lightweight and Efficient:** The backbone of the TTS Diffusion Transformer has only 0.45B parameters.
- 🎧**Ultra High-Quality Voice Cloning:** Zero-shot voice cloning with just a reference audio sample.
- 🌍**Bilingual Support:** Supports both Chinese and English, and code-switching.
- ✍️**Controllable:** Supports accent intensity control and fine-grained pronunciation adjustment.
- 💻**Windows Ready:** Fully compatible with Windows 10/11.

## Installation

### Prerequisites
- Windows 10/11
- **Conda (Miniconda or Anaconda)** - [Download here](https://docs.conda.io/en/latest/miniconda.html)
- NVIDIA GPU (recommended) or CPU
- Internet connection

### Option 1: One-Click Installer (Recommended)

**[📥 Download One-Click Installer](https://github.com/Saganaki22/MegaTTS3-WaveVAE/releases/tag/Installer)**

The easiest way to install MegaTTS3-WaveVAE:
- ✅ **Automated installation** of all dependencies
- ✅ **GPU Detection** with choice of NVIDIA 30xx/40xx/50xx series or CPU
- ✅ **One-click setup** for Windows x64

**Instructions:**
1. **Install Conda first** if you don't have it: [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html) (check "Add to PATH" during installation)
2. Download the installer from the [releases page](https://github.com/Saganaki22/MegaTTS3-WaveVAE/releases/tag/Installer)
3. **Unzip both files** in the folder where you want the project installed
4. **Run `install_megatts3_launcher.bat`** 
5. **Follow the on-screen instructions**
6. Select your GPU type when prompted (RTX 30xx/40xx/50xx or CPU)
7. Wait for installation to complete (~10-20 minutes)

The installer will automatically:
- Install Conda environment with Python 3.10
- Download and install all required dependencies
- Install the correct PyTorch version for your GPU
- Optionally download the model files
- Set up the project ready to run

### Option 2: Manual Installation

For advanced users or if you prefer manual control:

### Manual Installation Prerequisites
- Python 3.10
- Conda (Miniconda or Anaconda)
- Git

### Step 1: Install Conda
If you don't have Conda installed:
1. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for Windows
2. Install with "Add to PATH" option checked
3. Restart your terminal

### Step 2: Clone Repository
```bash
git clone https://github.com/Saganaki22/MegaTTS3-WaveVAE
cd MegaTTS3-WaveVAE
```

### Step 3: Create Python Environment
```bash
# Create Python 3.10 environment
conda create -n megatts3-env python=3.10
conda activate megatts3-env
```

### Step 4: Install Dependencies
```bash
# Install base requirements
pip install -r requirements.txt

# Install pynini through conda
conda install -y -c conda-forge pynini==2.1.5
```

### Step 5: Install PyTorch (Choose Your GPU)

**For RTX 30xx/40xx Series (CUDA 12.6):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**For RTX 50xx Series (CUDA 12.8):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**For CPU Only:**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 6: Install Additional Dependencies
```bash
# Install modelscope for model downloads
pip install modelscope

# Install FFmpeg if needed
conda install -c conda-forge ffmpeg
```

### Step 7: Set Environment Variables
**Option 1 - Permanent (Recommended):**
```bash
conda env config vars set PYTHONPATH="C:\path\to\MegaTTS3-WaveVAE;%PYTHONPATH%"
conda deactivate
conda activate megatts3-env
```

**Option 2 - Temporary:**
```bash
# Command Prompt
set PYTHONPATH="C:\path\to\MegaTTS3-WaveVAE;%PYTHONPATH%"

# PowerShell
$env:PYTHONPATH="C:\path\to\MegaTTS3-WaveVAE;$env:PYTHONPATH"
```

### Step 8: Download Models
```bash
# Download the MegaTTS model
modelscope download --model ACoderPassBy/MegaTTS-SFT --local_dir ./checkpoints
```

### Step 9: Verify Installation
```bash
# Test PyTorch GPU support
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Usage

### Web Interface (Recommended)
```bash
# Start the web UI
python tts/megatts3_gradio.py
```
Then open http://localhost:7929 in your browser.

### Command Line Interface
```bash
# Basic usage
python tts/infer_cli.py --input_wav 'path/to/reference.wav' --input_text "Your text here" --output_dir ./output

# With quality settings
python tts/infer_cli.py --input_wav 'path/to/reference.wav' --input_text "Your text here" --output_dir ./output --p_w 2.0 --t_w 3.0
```

### Parameters Explained
- `--p_w` (Intelligibility Weight): 1.0-5.0, higher = clearer pronunciation
- `--t_w` (Similarity Weight): 0.0-10.0, higher = more similar to reference voice
- For best results, set t_w 0-3 points higher than p_w

## Troubleshooting

### Common Issues

**1. "No module named 'tts'" error:**
```bash
# Make sure PYTHONPATH is set correctly
conda env config vars set PYTHONPATH="C:\your\path\to\MegaTTS3-WaveVAE"
conda deactivate
conda activate megatts3-env
```

**2. GPU not detected:**
- Ensure you installed the correct PyTorch version for your GPU
- Check CUDA drivers are installed and up to date

**3. FFmpeg errors:**
```bash
conda install -c conda-forge ffmpeg
```

**4. Audio file issues:**
- Use WAV format for best results
- Keep reference audio under 24 seconds
- Ensure good audio quality (clear speech, minimal noise)

## Model Information

This implementation uses the official MegaTTS3 model with:
- **Model Size**: 0.45B parameters
- **Supported Languages**: English, Chinese, and code-switching
- **Audio Format**: 24kHz WAV files
- **Max Reference Length**: 24 seconds

## Contributing

This is a Windows compatibility fork. For issues with the core model, please refer to the [original repository](https://github.com/bytedance/MegaTTS3). For Windows-specific issues, please open an issue in this repository.

## License

This project is licensed under the [Apache-2.0 License](LICENSE), same as the original MegaTTS3 project.

## Citation

If you use this work, please cite the original MegaTTS3 paper:

```bibtex
@article{jiang2025sparse,
  title={Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis},
  author={Jiang, Ziyue and Ren, Yi and Li, Ruiqi and Ji, Shengpeng and Ye, Zhenhui and Zhang, Chen and Jionghao, Bai and Yang, Xiaoda and Zuo, Jialong and Zhang, Yu and others},
  journal={arXiv preprint arXiv:2502.18924},
  year={2025}
}

@article{ji2024wavtokenizer,
  title={Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling},
  author={Ji, Shengpeng and Jiang, Ziyue and Wang, Wen and Chen, Yifu and Fang, Minghui and Zuo, Jialong and Yang, Qian and Cheng, Xize and Wang, Zehan and Li, Ruiqi and others},
  journal={arXiv preprint arXiv:2408.16532},
  year={2024}
}
```

## Acknowledgments

- Original MegaTTS3 by [ByteDance](https://github.com/bytedance/MegaTTS3)
- WaveVAE model provided by [ACoderPassBy/MegaTTS-SFT](https://modelscope.cn/models/ACoderPassBy/MegaTTS-SFT/summary) and [mrfakename/MegaTTS3-VoiceCloning](https://huggingface.co/mrfakename/MegaTTS3-VoiceCloning/tree/main)
- Windows compatibility improvements by [Saganaki22](https://github.com/Saganaki22)
