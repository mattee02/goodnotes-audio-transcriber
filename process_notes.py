#!/usr/bin/env python3
"""
GoodNotes Audio Processor & Transcriber
========================================
A general-purpose tool to extract recorded audio from GoodNotes notes
(.goodnotes, .goodnotes.zip, .zip, or uncompressed directories),
remove background noise using DeepFilterNet, and transcribe speech
using faster-whisper.

Pipeline:
  1. Direct audio extraction to a temporary directory without unpacking
     the entire archive onto disk.
  2. Background noise suppression (DeepFilterNet) with chunked streaming
     to avoid memory spikes.
  3. Speech-to-text transcription (faster-whisper) with Voice Activity
     Detection (VAD).
  4. Output generation in clean, readable formats (MP3 audio by default,
     plain text transcripts) with automatic cleanup of intermediate data.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path


# ==============================================================================
# Dynamic Compatibility Shims
# ==============================================================================

def apply_compatibility_patches():
    """
    Apply runtime shims for upstream library deprecations:
    1. torchaudio >= 2.1: removed 'torchaudio.backend', which breaks DeepFilterNet imports.
    2. PyAV >= 18: removed 'metadata_errors' parameter from av.open(), which breaks
       certain faster-whisper versions.
    """
    # 1. torchaudio backend shim
    try:
        import torchaudio.backend.common  # type: ignore # noqa: F401
    except (ImportError, ModuleNotFoundError, AttributeError):
        try:
            import torchaudio

            backend = types.ModuleType("torchaudio.backend")
            common = types.ModuleType("torchaudio.backend.common")

            class AudioMetaData:
                def __init__(self, sample_rate=0, num_frames=0, num_channels=0, bits_per_sample=0, encoding=""):
                    self.sample_rate = sample_rate
                    self.num_frames = num_frames
                    self.num_channels = num_channels
                    self.bits_per_sample = bits_per_sample
                    self.encoding = encoding

            common.AudioMetaData = getattr(torchaudio, "AudioMetaData", AudioMetaData)
            backend.common = common
            torchaudio.backend = backend
            sys.modules["torchaudio.backend"] = backend
            sys.modules["torchaudio.backend.common"] = common
        except ImportError:
            pass

    # 2. PyAV metadata_errors shim
    try:
        import av

        orig_open = av.open

        def patched_av_open(*args, **kwargs):
            kwargs.pop("metadata_errors", None)
            return orig_open(*args, **kwargs)

        av.open = patched_av_open
    except Exception:
        pass


apply_compatibility_patches()


# ==============================================================================
# Helper Functions and GoodNotes Discovery
# ==============================================================================

def get_clean_name(path: Path) -> str:
    """Return a clean note name by stripping common GoodNotes/zip extensions."""
    name = path.name
    for ext in (".goodnotes.zip", ".goodnotes", ".zip"):
        if name.lower().endswith(ext):
            return name[:-len(ext)]
    return path.stem


def is_goodnotes_archive(path: Path) -> bool:
    """Check if a zip file is a valid GoodNotes document archive."""
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            return any(n.startswith("attachments/") or n.endswith(".pb") for n in names)
    except (zipfile.BadZipFile, OSError):
        return False


def find_notes(inputs: list[str] | None, search_dirs: list[Path]) -> list[Path]:
    """
    Find GoodNotes notes to process.
    If specific input paths are provided, use them; otherwise scan search directories.
    """
    if inputs:
        results = []
        for inp in inputs:
            p = Path(inp)
            if p.exists():
                results.append(p)
            else:
                # Try relative to search dirs
                found = False
                for sdir in search_dirs:
                    candidate = sdir / inp
                    if candidate.exists():
                        results.append(candidate)
                        found = True
                        break
                if not found:
                    print(f"[!] Warning: path not found: {inp}", file=sys.stderr)
        return results

    # Automatic discovery
    candidates = []
    ignored_dir_names = {
        "to_filter", "filtered", "transcripts", "audio",
        ".venv", ".git", "figures", "__pycache__", ".agents"
    }

    seen_stems = set()
    for sdir in search_dirs:
        if not sdir.is_dir():
            continue
        for item in sorted(sdir.iterdir()):
            if item.is_file():
                name_lower = item.name.lower()
                if name_lower.endswith((".goodnotes", ".goodnotes.zip", ".zip")):
                    if is_goodnotes_archive(item):
                        stem = get_clean_name(item)
                        if stem not in seen_stems:
                            candidates.append(item)
                            seen_stems.add(stem)
            elif item.is_dir():
                if item.name in ignored_dir_names:
                    continue
                if (item / "attachments").is_dir():
                    stem = get_clean_name(item)
                    if stem not in seen_stems:
                        candidates.append(item)
                        seen_stems.add(stem)

    return sorted(candidates, key=lambda p: p.name)


def is_already_processed(clean_name: str, transcripts_dir: Path) -> bool:
    """Check if a transcript already exists for this note."""
    note_dir = transcripts_dir / clean_name
    if not note_dir.is_dir():
        return False
    txt_files = [f for f in note_dir.glob("*.txt") if f.is_file() and f.stat().st_size > 0]
    return len(txt_files) > 0


# ==============================================================================
# Media Extraction via FFmpeg / FFprobe
# ==============================================================================

def check_external_tools():
    """Verify that required system utilities (ffmpeg, ffprobe) are installed."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            print(f"[ERROR] Required tool '{tool}' was not found in PATH.", file=sys.stderr)
            print("Please install FFmpeg using your system package manager.", file=sys.stderr)
            sys.exit(1)


def is_audio_file(file_path: Path) -> bool:
    """Check whether a file contains a readable audio stream using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return bool(res.stdout.strip())
    except Exception:
        return False


def convert_to_mono_48k_wav(input_path: Path, output_wav: Path) -> bool:
    """Convert an audio/video file to 48000Hz mono WAV (required by DeepFilterNet)."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(input_path),
        "-ar", "48000",
        "-ac", "1",
        str(output_wav)
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Error converting {input_path.name} to WAV: {e}", file=sys.stderr)
        return False


def extract_audio_attachments(source: Path, temp_dir: Path) -> list[tuple[str, Path]]:
    """
    Extract audio attachments from a GoodNotes archive or folder into temp_dir.
    Returns list of tuples: (attachment_id, temp_wav_path).
    """
    tracks: list[tuple[str, Path]] = []

    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source, "r") as z:
            attachment_names = [n for n in z.namelist() if "attachments/" in n and not n.endswith("/")]
            for member in sorted(attachment_names):
                extracted = Path(z.extract(member, temp_dir))
                att_id = extracted.name
                if is_audio_file(extracted):
                    wav_temp = temp_dir / f"{att_id}_raw.wav"
                    if convert_to_mono_48k_wav(extracted, wav_temp):
                        tracks.append((att_id, wav_temp))
                # Delete raw attachment file immediately to preserve disk space
                extracted.unlink(missing_ok=True)

    elif source.is_dir():
        att_dir = source / "attachments" if (source / "attachments").is_dir() else source
        for filepath in sorted(att_dir.rglob("*")):
            if filepath.is_file() and is_audio_file(filepath):
                att_id = filepath.name
                wav_temp = temp_dir / f"{att_id}_raw.wav"
                if convert_to_mono_48k_wav(filepath, wav_temp):
                    tracks.append((att_id, wav_temp))

    return tracks


# ==============================================================================
# Noise Suppression (DeepFilterNet3)
# ==============================================================================

def load_deepfilternet(device: str = "auto"):
    """Initialize and load DeepFilterNet3 model in memory."""
    print("Initializing DeepFilterNet3...")
    try:
        from df.enhance import init_df
        model, df_state, _ = init_df()
        return model, df_state
    except Exception as e:
        print(f"[ERROR] Failed to initialize DeepFilterNet: {e}", file=sys.stderr)
        sys.exit(1)


def denoise_audio_file(
    df_model,
    df_state,
    input_wav: Path,
    output_path: Path,
    output_format: str = "mp3"
) -> Path:
    """
    Process audio with DeepFilterNet3 in 3-minute chunks to prevent memory spikes.
    Saves output in the requested format (MP3, WAV, or M4A).
    """
    import soundfile as sf
    import torch
    from df.enhance import enhance

    SR = 48000
    CHUNK_SAMPLES = 180 * SR  # 3 minutes at 48kHz
    temp_wav = output_path.with_suffix(".tmp.wav")

    with sf.SoundFile(str(input_wav)) as src:
        total_samples = len(src)
        sr = src.samplerate
        channels = src.channels

        with sf.SoundFile(
            str(temp_wav), mode="w", samplerate=sr, channels=channels, subtype="PCM_16"
        ) as dst:
            processed_samples = 0
            while True:
                block = src.read(CHUNK_SAMPLES, dtype="float32")
                if len(block) == 0:
                    break

                if block.ndim == 1:
                    tensor = torch.from_numpy(block).unsqueeze(0)
                else:
                    tensor = torch.from_numpy(block.T)

                with torch.no_grad():
                    enhanced = enhance(df_model, df_state, tensor)

                enhanced_np = enhanced.squeeze(0).cpu().numpy()
                dst.write(enhanced_np)

                processed_samples += len(block)
                pct = min(100.0, (processed_samples / max(1, total_samples)) * 100)
                dur_proc = processed_samples // sr
                dur_tot = total_samples // sr
                print(
                    f"     [Denoise] {pct:5.1f}% completed ({dur_proc // 60:02d}:{dur_proc % 60:02d} / {dur_tot // 60:02d}:{dur_tot % 60:02d})",
                    end="\r",
                    flush=True,
                )
            print()

    output_format = output_format.lower()
    if output_format == "mp3":
        final_file = output_path.with_suffix(".mp3")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(temp_wav), "-c:a", "libmp3lame", "-b:a", "128k", str(final_file)]
        subprocess.run(cmd, check=True)
        temp_wav.unlink(missing_ok=True)
        return final_file
    elif output_format in ("m4a", "aac"):
        final_file = output_path.with_suffix(".m4a")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(temp_wav), "-c:a", "aac", "-b:a", "128k", str(final_file)]
        subprocess.run(cmd, check=True)
        temp_wav.unlink(missing_ok=True)
        return final_file
    else:
        final_file = output_path.with_suffix(".wav")
        if temp_wav != final_file:
            temp_wav.replace(final_file)
        return final_file


# ==============================================================================
# Speech-to-Text Transcription (faster-whisper)
# ==============================================================================

def load_whisper(model_size: str = "base", device: str = "auto"):
    """Initialize and load faster-whisper model."""
    print(f"Loading faster-whisper (model: '{model_size}', device: '{device}')...")
    try:
        from faster_whisper import WhisperModel
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load faster-whisper: {e}", file=sys.stderr)
        sys.exit(1)


def transcribe_file(whisper_model, audio_path: Path, output_txt: Path, language: str = "en") -> Path:
    """Transcribe audio file and write timestamped segments to output_txt."""
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    lang_param = None if language.lower() in ("auto", "none") else language

    segments, _ = whisper_model.transcribe(
        str(audio_path),
        beam_size=5,
        language=lang_param,
        vad_filter=True,
    )

    with open(output_txt, "w", encoding="utf-8") as out_f:
        for seg in segments:
            out_f.write(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text.strip()}\n")

    return output_txt


# ==============================================================================
# Single Note Processing
# ==============================================================================

def process_single_note(
    source: Path,
    df_model,
    df_state,
    whisper_model,
    language: str,
    audio_dir: Path,
    transcripts_dir: Path,
    skip_filter: bool,
    audio_format: str,
    no_audio: bool,
    force: bool,
) -> bool:
    """Run the complete extraction, denoising, and transcription pipeline on one note."""
    clean_name = get_clean_name(source)

    if not force and is_already_processed(clean_name, transcripts_dir):
        print(f"[-] Skipping '{clean_name}': already transcribed (use --force to overwrite).")
        return False

    print(f"\n{'='*70}")
    print(f"  Processing: {clean_name} ({source.name})")
    print(f"{'='*70}")

    start_time = time.time()

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        print("  1. Extracting audio attachments...")
        tracks = extract_audio_attachments(source, temp_dir)

        if not tracks:
            print(f"  [!] No audio tracks found in {source.name}.")
            return False

        print(f"     Found {len(tracks)} audio track(s).")

        for idx, (_, raw_wav) in enumerate(tracks, start=1):
            suffix = f"_part{idx}" if len(tracks) > 1 else ""
            track_name = f"{clean_name}{suffix}"

            note_audio_dir = audio_dir / clean_name
            note_transcript_dir = transcripts_dir / clean_name
            note_audio_dir.mkdir(parents=True, exist_ok=True)
            note_transcript_dir.mkdir(parents=True, exist_ok=True)

            target_audio_base = note_audio_dir / track_name
            target_transcript = note_transcript_dir / f"{track_name}.txt"

            # 2. Denoising
            if skip_filter:
                print("  2. [Filter skipped] Using raw extracted audio...")
                if audio_format.lower() == "mp3":
                    clean_audio_path = target_audio_base.with_suffix(".mp3")
                    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(raw_wav), "-c:a", "libmp3lame", "-b:a", "128k", str(clean_audio_path)]
                    subprocess.run(cmd, check=True)
                else:
                    clean_audio_path = target_audio_base.with_suffix(".wav")
                    shutil.copy2(raw_wav, clean_audio_path)
            else:
                print("  2. Removing background noise with DeepFilterNet3...")
                clean_audio_path = denoise_audio_file(
                    df_model, df_state, raw_wav, target_audio_base, output_format=audio_format
                )
                size_mb = clean_audio_path.stat().st_size / (1024 * 1024)
                print(f"     Clean audio: {clean_audio_path.name} ({size_mb:.1f} MB)")

            # 3. Transcription
            print(f"  3. Transcribing with faster-whisper [language: {language}]...")
            transcribe_file(whisper_model, clean_audio_path, target_transcript, language=language)
            print(f"     Transcript saved: {target_transcript}")

            if no_audio:
                clean_audio_path.unlink(missing_ok=True)
                print("     Clean audio removed as requested (--no-audio).")

    elapsed = time.time() - start_time
    print(f"  Finished in {int(elapsed // 60)}m {int(elapsed % 60)}s!")
    return True


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract, denoise, and transcribe audio recordings from GoodNotes documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Optional specific .goodnotes, .goodnotes.zip files or directories to process.",
    )
    parser.add_argument(
        "--input-dir",
        default="inputs",
        help="Dedicated directory containing input GoodNotes archives.",
    )
    parser.add_argument(
        "-l", "--language",
        default="en",
        help="Spoken language code for Whisper (e.g. 'en', 'it', 'auto').",
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        help="Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3').",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip DeepFilterNet noise suppression.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device ('auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--audio-dir",
        default="filtered",
        help="Destination directory for cleaned audio files.",
    )
    parser.add_argument(
        "--transcripts-dir",
        default="transcripts",
        help="Destination directory for text transcripts.",
    )
    parser.add_argument(
        "--format",
        default="mp3",
        choices=["mp3", "m4a", "wav"],
        help="Audio format for saved clean audio (MP3 saves ~90%% space with high speech quality).",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Discard clean audio after transcription (keeps only .txt transcripts).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if transcript already exists.",
    )

    args = parser.parse_args()

    check_external_tools()

    root_dir = Path.cwd()
    input_dir_path = root_dir / args.input_dir

    search_dirs = [input_dir_path, root_dir] if input_dir_path.exists() else [root_dir]
    notes = find_notes(args.inputs, search_dirs)

    if not notes:
        print("[!] No GoodNotes notes found to process.")
        print(f"    Place your .goodnotes or .goodnotes.zip files in '{args.input_dir}/' or specify paths directly.")
        return

    transcripts_dir = Path(args.transcripts_dir)
    audio_dir = Path(args.audio_dir)

    to_process = [n for n in notes if args.force or not is_already_processed(get_clean_name(n), transcripts_dir)]

    if not to_process:
        print("[+] All detected notes are already transcribed!")
        print("    Use --force to reprocess them if needed.")
        return

    print(f"[+] Found {len(to_process)} note(s) to process:")
    for n in to_process:
        print(f"    - {n.name}")

    df_model, df_state = (None, None)
    if not args.skip_filter:
        df_model, df_state = load_deepfilternet(device=args.device)

    whisper_model = load_whisper(model_size=args.model, device=args.device)

    processed_count = 0
    for note in to_process:
        ok = process_single_note(
            source=note,
            df_model=df_model,
            df_state=df_state,
            whisper_model=whisper_model,
            language=args.language,
            audio_dir=audio_dir,
            transcripts_dir=transcripts_dir,
            skip_filter=args.skip_filter,
            audio_format=args.format,
            no_audio=args.no_audio,
            force=args.force,
        )
        if ok:
            processed_count += 1

    print(f"\n{'='*70}")
    print(f"[✓] Completed processing! {processed_count} note(s) processed.")
    print(f"    Transcripts: {transcripts_dir.resolve()}")
    if not args.no_audio:
        print(f"    Clean audio: {audio_dir.resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
