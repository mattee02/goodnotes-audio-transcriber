# GoodNotes Audio Transcriber

An automated, end-to-end command-line tool that extracts audio recordings from **GoodNotes** documents, suppresses background ambient noise using deep learning (**DeepFilterNet3**), and transcribes speech into timestamped text using **faster-whisper**.

---

## 🎯 The Problem

[GoodNotes](https://www.goodnotes.com/) is widely used across iPad, Mac, and iOS to capture audio recordings during lectures, meetings, and interviews alongside handwritten notes. However, processing these recordings presents several practical challenges:

1. **Bundled Media**: Audio recordings are embedded inside GoodNotes document packages (`.goodnotes` or `.zip`) alongside document data, requiring extraction and format conversion before external tools or media players can use them.
2. **Room Acoustics & Background Noise**: Audio captured in lecture halls and meeting rooms frequently suffers from reverberation, air-conditioning hum, keyboard clatter, and ambient room noise.
3. **Manual Multi-Step Overhead**: Extracting audio tracks, running speech-enhancement algorithms, and feeding clean streams into speech-to-text models typically requires stitching together multiple tools, creating gigabytes of temporary uncompressed audio files.

**GoodNotes Audio Transcriber** automates this entire lifecycle into a single, clean command.

---

## 🔄 How It Works

```text
┌─────────────────────────┐
│  GoodNotes Documents    │   (.goodnotes, .goodnotes.zip, .zip)
│  placed in inputs/      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 1. Direct Inspection    │   Inspects archives on the fly via FFprobe / FFmpeg
│    & Extraction         │   Converts audio streams to 48kHz mono in memory/temp
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 2. AI Speech            │   Applies DeepFilterNet3 in streaming 3-minute chunks
│    Enhancement          │   Removes room acoustics, AC rumble, and background noise
└────────────┬────────────┘
             │
             ├──────────────────────────────────────────────┐
             ▼                                              ▼
┌─────────────────────────┐                    ┌─────────────────────────┐
│ 3. Audio Optimization   │                    │ 4. AI Transcription     │
│    Saves clean MP3      │                    │    faster-whisper (STT) │
│    (~90% disk savings)  │                    │    with VAD filtering   │
└────────────┬────────────┘                    └────────────┬────────────┘
             │                                              │
             ▼                                              ▼
      filtered/<name>/                              transcripts/<name>/
         <name>.mp3                                     <name>.txt
```

---

## ✨ Features

- **Direct Archive Processing**: Ingests `.goodnotes` and `.goodnotes.zip` files directly without requiring you to unzip or modify original documents.
- **Deep Learning Denoising**: Uses [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) to isolate human voice and eliminate background noise. Processes audio in streaming chunks to maintain a low RAM footprint regardless of recording length.
- **Fast, Accurate Transcription**: Leverages [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 reimplementation of OpenAI Whisper) with built-in Voice Activity Detection (VAD) for fast, highly accurate speech-to-text.
- **Optimized Storage**: Encodes enhanced audio to 128 kbps MP3 by default, reducing file size by ~90% compared to uncompressed WAV while preserving full vocal clarity.
- **Zero Disk Clutter**: Temporary conversion files exist only during processing and are immediately purged.
- **Smart Incremental Runs**: Keeps track of processed files and skips notes that have already been transcribed, making subsequent runs instant.
- **Human-Friendly Naming**: Transcripts and clean audio files are automatically named after the note (e.g. `lecture_05.txt`, `lecture_05.mp3`).

---

## 📌 Prerequisites

1. **Python 3.10+** (Python 3.11 is recommended for pre-built binary wheels).
2. **FFmpeg & FFprobe**: Required for audio transcoding and stream probing.
   - **Fedora / RHEL**: `sudo dnf install ffmpeg`
   - **Ubuntu / Debian**: `sudo apt install ffmpeg`
   - **macOS**: `brew install ffmpeg`

### 🦀 Note on Rust (`cargo`)
`DeepFilterNet` relies on native extensions (`deepfilterlib` in Rust).
- On **Python 3.10 and 3.11**, precompiled binary wheels are downloaded automatically from PyPI.
- On **Python 3.12+**, `pip` builds the extension from source and requires the **Rust / Cargo** toolchain.
  ```bash
  # Install Rust via rustup if needed:
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

---

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd goodnotes-audio-transcriber
   ```

2. **Add your GoodNotes files**:
   Place your `.goodnotes` or `.goodnotes.zip` files into the `inputs/` directory:
   ```bash
   cp /path/to/my_lecture.goodnotes inputs/
   ```

3. **Run the pipeline**:
   ```bash
   ./process_notes.sh
   ```
   The wrapper script automatically sets up a Python virtual environment (`.venv`), verifies dependencies, checks system tools, and processes all notes found in `inputs/`.

---

## 📖 Usage & Examples

### Basic Run (Scans `inputs/`)
```bash
./process_notes.sh
```

### Process Specific Files
You can target one or more specific notes directly:
```bash
./process_notes.sh inputs/meeting_2026-10-27.goodnotes
```

### Change Language
Whisper defaults to English (`en`). To transcribe in Italian, Spanish, or other languages:
```bash
./process_notes.sh --language it
# Or enable automatic language detection:
./process_notes.sh --language auto
```

### Select Whisper Model Size
Balance speed versus accuracy by choosing a model (`tiny`, `base`, `small`, `medium`, `large-v3`):
```bash
./process_notes.sh --model small
```

### Skip Denoising (Faster Processing)
If the audio was recorded with a close-proximity microphone and does not require filtering:
```bash
./process_notes.sh --skip-filter
```

### Text Transcripts Only
If you do not need to keep the cleaned audio files and only want `.txt` transcripts:
```bash
./process_notes.sh --no-audio
```

### Force Reprocessing
By default, already transcribed notes are skipped. To re-process existing notes:
```bash
./process_notes.sh --force
```

---

## 📂 Output Directory Structure

After processing, your outputs are organized as follows:

```text
├── inputs/                    <- Source GoodNotes files (kept intact)
│   ├── lecture_01.goodnotes.zip
│   └── lecture_02.goodnotes
├── filtered/                  <- Enhanced, denoised MP3 audio files
│   ├── lecture_01/
│   │   └── lecture_01.mp3
│   └── lecture_02/
│       └── lecture_02.mp3
└── transcripts/               <- Sized, timestamped plain text transcripts
    ├── lecture_01/
    │   └── lecture_01.txt
    └── lecture_02/
        └── lecture_02.txt
```

### Sample Transcript Output (`transcripts/lecture_01/lecture_01.txt`)
```text
[0.00s -> 5.12s] Welcome everyone. Today we are discussing Systems Security Engineering.
[5.12s -> 12.40s] Before we begin, let's examine the difference between design intent and user intent.
[12.40s -> 19.85s] As formalized in the NIST guidelines, security represents the preservation of authorized behaviors.
```

---

## ⚙️ CLI Reference

| Flag / Option | Description | Default |
| :--- | :--- | :--- |
| `inputs` | Optional specific file(s) or folder(s) to process | Auto-discovery |
| `--input-dir` | Directory to scan for GoodNotes archives | `inputs` |
| `-l`, `--language` | Language code for Whisper transcription (`en`, `it`, `auto`, etc.) | `en` |
| `-m`, `--model` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) | `base` |
| `--format` | Output audio format for enhanced audio (`mp3`, `m4a`, `wav`) | `mp3` |
| `--skip-filter` | Skip DeepFilterNet background noise suppression | `False` |
| `--no-audio` | Discard clean audio after transcription (saves only `.txt`) | `False` |
| `--force` | Force reprocessing of already transcribed notes | `False` |
| `--audio-dir` | Output directory for enhanced audio | `filtered` |
| `--transcripts-dir`| Output directory for text transcripts | `transcripts` |
| `--device` | Inference execution device (`auto`, `cuda`, `cpu`) | `auto` |

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
