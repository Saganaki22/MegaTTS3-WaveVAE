<div align="center">
    <h1>
    MegaTTS3-WaveVAE <img src="./assets/fig/Hi.gif" width="40px">
    </h1>
    <p>
    Unofficial Windows-Compatible Implementation with Voice Library & Audiobook Studio<br>
    </p>
</div>
<div align="center">
    <a href="https://github.com/Saganaki22/MegaTTS3-WaveVAE"><img src="https://img.shields.io/badge/GitHub-Repository-black?logo=github" alt="GitHub"></a>
    <a href="https://huggingface.co/drbaph/MegaTTS3-WaveVAE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="Hugging Face Model"></a>
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

<img width="1932" height="1265" alt="image" src="https://github.com/user-attachments/assets/eac17cee-743b-4c8d-84ad-ffd04d997d73" />


## About This Fork

This is an **unofficial Windows-compatible fork** of the original [ByteDance MegaTTS3](https://github.com/bytedance/MegaTTS3) repository. This version includes:

- ✅ **Windows Compatibility**: Pre-configured for Windows installation
- ✅ **GPU Support**: Optimized PyTorch installations for RTX 30xx/40xx and RTX 50xx series
- ✅ **Simplified Setup**: Streamlined installation process with proper dependencies
- ✅ **Enhanced Web UI**: Comprehensive Gradio interface with Voice Library & Audiobook Studio
- ✅ **WaveVAE Included**: Includes the WaveVAE encoder/decoder thanks to [ACoderPassBy/MegaTTS-SFT](https://modelscope.cn/models/ACoderPassBy/MegaTTS-SFT)
- 🎭 **Voice Library System**: Create, save, and manage voice profiles with custom settings
- 📚 **Audiobook Creation**: Generate single-voice and multi-character audiobooks with smart chunking
- 🚀 **Batch Processing**: Optimized multi-processing for faster audiobook generation

## Key Features

### Core TTS Features
- 🚀**Lightweight and Efficient:** The backbone of the TTS Diffusion Transformer has only 0.45B parameters.
- 🎧**Ultra High-Quality Voice Cloning:** Zero-shot voice cloning with just a reference audio sample.
- 🌍**Bilingual Support:** Supports both Chinese and English, and code-switching.
- ✍️**Controllable:** Supports accent intensity control and fine-grained pronunciation adjustment.
- 💻**Windows Ready:** Fully compatible with Windows 10/11.

### New Voice Library & Audiobook Features
- 🎭 **Voice Profile Management**: Save and organize voice profiles with custom names, descriptions, and settings
- 📝 **Smart Text Processing**: Automatic text chunking at sentence boundaries for optimal processing
- 📚 **Single-Voice Audiobooks**: Create consistent audiobooks with one narrator voice
- 🎭 **Multi-Voice Audiobooks**: Support for multiple character voices using voice tags `[character_name]`
- 🚀 **Batch Processing**: Optimized parallel processing for faster audiobook generation
- 📁 **Project Management**: Organized output with project folders and numbered audio chunks
- 🔄 **Persistent Settings**: Voice library path and configurations saved between sessions
- 📄 **File Upload Support**: Load text from .txt, .md, and .rtf files for audiobook creation

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

**Option A: Via ModelScope (Recommended)**
```bash
# Download the MegaTTS model via ModelScope
modelscope download --model ACoderPassBy/MegaTTS-SFT --local_dir ./checkpoints
```

**Option B: Direct Download from Hugging Face**
Alternatively, you can download the model files directly from our Hugging Face repository:

1. Visit: [https://huggingface.co/drbaph/MegaTTS3-WaveVAE/tree/main](https://huggingface.co/drbaph/MegaTTS3-WaveVAE/tree/main)
2. Download all the model files and folders
3. Place them in the `./checkpoints/` folder in your project directory

The checkpoints folder should contain:
- `diffusion_transformer/`
- `wavvae/` 
- `g2p/`
- `aligner_lm/`
- `configuration.json`
- `config.json`

### Step 9: Verify Installation
```bash
# Test PyTorch GPU support
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Usage

### Web Interface (Recommended)
```bash
# Start the Voice Library & Audiobook Studio
python megatts3_gradio.py
```
Then open http://localhost:7929 in your browser.

The enhanced web interface includes four main tabs:

#### 🎤 Text-to-Speech
- Choose from saved voice profiles or upload reference audio
- Real-time voice selection with automatic settings loading
- Advanced parameter controls for fine-tuning

#### 📚 Voice Library
- **Create Voice Profiles**: Save reference audio with custom names and descriptions
- **Manage Settings**: Store inference timesteps, intelligibility, and similarity weights per voice
- **Test Voices**: Preview voices before saving with custom test text
- **Organize Profiles**: View, edit, and delete voice profiles with rich metadata

#### 📖 Audiobook Creation - Single Voice
- **Text Input**: Paste text or upload .txt/.md/.rtf files
- **Smart Chunking**: Automatic text splitting at sentence boundaries (~50 words per chunk)
- **Voice Selection**: Choose from your saved voice library
- **Project Management**: Organized output with project folders
- **Batch Processing**: Efficient parallel generation for long texts

#### 🎭 Audiobook Creation - Multi-Voice
- **Voice Tagging**: Use `[character_name]` tags to assign text to different voices
- **Character Management**: Automatic detection and validation of voice assignments
- **Batch Processing**: Optimized parallel processing for multiple voices
- **Project Organization**: Files named with character information (e.g., `project_001_narrator.wav`)

### Voice Library Management

#### Creating Voice Profiles
1. Go to the **Voice Library** tab
2. Set your voice library folder path (saved automatically)
3. Fill in voice details:
   - **Voice Name**: Internal identifier (e.g., `deep_male_narrator`)
   - **Display Name**: User-friendly name (e.g., `Deep Male Narrator`)
   - **Description**: Optional details about the voice
4. Upload reference audio (10-25 seconds of clear speech)
5. Adjust TTS parameters (timesteps, intelligibility, similarity weights)
6. Test the voice with sample text
7. Save the profile for reuse

#### Using Saved Voices
- In TTS tab: Select from the voice dropdown
- Settings automatically load (timesteps, weights)
- Reference audio is used automatically
- Manual audio upload is hidden when using saved voices

### Audiobook Creation Workflow

#### Single-Voice Audiobooks
1. Go to **Audiobook Creation - Single Voice** tab
2. Add your text (paste or upload file)
3. Select a voice from your library
4. Set a project name
5. Click **Validate Input** to check everything
6. Click **Create Audiobook** to generate

#### Multi-Voice Audiobooks
1. Go to **Audiobook Creation - Multi-Voice** tab
2. Format your text with voice tags:
   ```
   [narrator] Once upon a time, in a land far away...
   [princess] "Hello there!" she said cheerfully.
   [narrator] The mysterious figure walked away.
   [wizard] "You shall not pass!" he declared.
   ```
3. Set a project name
4. Click **Validate Text & Voices** to check all voices exist
5. Click **Create Multi-Voice Audiobook** to generate

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

## Voice Library System

### Configuration Management
- **Persistent Settings**: Voice library path saved in `megatts3_config.json`
- **Voice Profiles**: Each voice stored in separate folder with `config.json`
- **Automatic Loading**: Voice settings automatically applied when selected

### File Structure
```
voice_library/
├── deep_male_narrator/
│   ├── config.json           # Voice metadata and settings
│   └── reference.wav         # Reference audio file
├── female_character/
│   ├── config.json
│   └── reference.mp3
└── ...
```

### Voice Profile Format
```json
{
  "display_name": "Deep Male Narrator",
  "description": "Authoritative voice for main character",
  "audio_file": "reference.wav",
  "infer_timestep": 32,
  "p_w": 1.4,
  "t_w": 3.0,
  "created_date": "1640995200.0",
  "version": "1.0"
}
```

## Audiobook Features

### Smart Text Chunking
- **Sentence Boundary Detection**: Splits at natural sentence endings
- **Word Count Optimization**: ~50 words per chunk for optimal processing
- **Punctuation Handling**: Handles multiple punctuation marks (. ! ?)
- **Character Limit**: Minimum 10 characters per text input

### Multi-Voice Processing
- **Voice Tag System**: Use `[voice_name]` to assign text to characters
- **Character Validation**: Ensures all referenced voices exist in library
- **Automatic Cleanup**: Removes character names from text if they match voice tags
- **Batch Generation**: Parallel processing for multiple voices

### Project Organization
- **Project Folders**: All output organized in `audiobook_projects/project_name/`
- **Numbered Chunks**: Files named sequentially (001, 002, 003...)
- **Character Identification**: Multi-voice files include character names
- **Combined Preview**: Full audiobook preview in web interface

### Output Examples
**Single Voice:**
```
audiobook_projects/my_story/
├── my_story_001.wav
├── my_story_002.wav
└── my_story_003.wav
```

**Multi-Voice:**
```
audiobook_projects/dialogue_story/
├── dialogue_story_001_narrator.wav
├── dialogue_story_002_princess.wav
├── dialogue_story_003_narrator.wav
└── dialogue_story_004_wizard.wav
```

## Troubleshooting

### Common Issues

**1. "No module named 'tts'" error:**
```bash
# Make sure PYTHONPATH is set correctly
conda env config vars set PYTHONPATH="C:\your\path\to\MegaTTS3-WaveVAE"
conda deactivate
conda activate megatts3-env
```

**2. AssertionError: | ckpt not found**
This means the model files haven't been downloaded yet. Choose one of these methods:

**Method A: Via ModelScope**
```bash
# Activate the environment
conda activate megatts3-env

# Navigate to your project directory
cd path\to\MegaTTS3-WaveVAE

# Download the model
modelscope download --model ACoderPassBy/MegaTTS-SFT --local_dir ./checkpoints
```

**Method B: Direct Download from Hugging Face**
1. Visit: [https://huggingface.co/drbaph/MegaTTS3-WaveVAE/tree/main](https://huggingface.co/drbaph/MegaTTS3-WaveVAE/tree/main)
2. Download all model files and folders
3. Place them in the `./checkpoints/` folder in your project directory

**3. GPU not detected:**
- Ensure you installed the correct PyTorch version for your GPU
- Check CUDA drivers are installed and up to date

**4. FFmpeg errors:**
```bash
conda install -c conda-forge ffmpeg
```

**5. Audio file issues:**
- Use WAV format for best results
- Keep reference audio under 24 seconds
- Ensure good audio quality (clear speech, minimal noise)

**6. Voice Library Issues:**
- **Voice not found**: Check that voice profiles are properly saved in the voice library folder
- **Config errors**: Ensure `config.json` files are properly formatted
- **Path issues**: Use full absolute paths for voice library folder
- **Permission errors**: Ensure write access to voice library and project folders

**7. Audiobook Generation Issues:**
- **Empty chunks**: Text may be too short after cleaning - use longer sentences
- **Voice validation fails**: Ensure all `[voice_name]` tags match saved voice profile names exactly
- **Memory issues**: For very long texts, consider breaking into smaller projects
- **Batch processing slow**: Reduce batch size or use fewer worker processes

## Advanced Configuration

### Batch Processing Settings
The system uses optimized batch processing for audiobook generation:
- **Default Batch Size**: 3 chunks per batch
- **Worker Processes**: 1 worker by default (adjustable)
- **GPU Memory**: Automatically managed per batch

### Performance Optimization
- **GPU Utilization**: Batched processing maximizes GPU efficiency
- **Memory Management**: Automatic cleanup between batches
- **Parallel Processing**: Multiple chunks processed simultaneously
- **Smart Queuing**: Optimized task distribution

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
