#!/usr/bin/env python

# OpenWakeWord "hey jarvis" + NVIDIA Parakeet TDT 0.6B v2 demo.
#
# Behavior:
#   1. Listens continuously on the mic.
#   2. Uses openWakeWord to detect ONLY the "hey jarvis" wakeword.
#   3. When "hey jarvis" fires (first score spike above wake_threshold),
#      records your speech until end-of-utterance (silence) or a max length.
#   4. Sends that *single* audio chunk to nvidia/parakeet-tdt-0.6b-v2 (NeMo ASR)
#      and prints the transcript.
#
# Run, for example:
#   uv run detect_from_microphone.py --inference_framework onnx
#
# Optional:
#   --model_path /path/to/my_jarvis.onnx
#   --wake_threshold 0.95
#   --silence_threshold 500
#   --silence_duration 1.2
#   --max_utterance 10.0
#   --asr_device cuda   (or cpu)

import argparse
import os
import sys
import tempfile
import cuda
import wave
from typing import Optional, List

import numpy as np
import pyaudio
from openwakeword.model import Model

# Audio / wake word defaults
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # matches Parakeet's expected input


def find_hey_jarvis_model(inference_framework: str) -> str:
    """
    Locate the built-in 'hey jarvis' model shipped with openWakeWord.

    This assumes you have already run `openwakeword.utils.download_models()`
    at least once in this environment (in many installs, the models are
    already bundled with the package).
    """
    try:
        import importlib.resources as pkg_resources
        import openwakeword  # noqa: F401
        from pathlib import Path  # noqa: F401

        # Models live under openwakeword/resources/models
        models_dir = pkg_resources.files("openwakeword") / "resources" / "models"
        suffix = ".onnx" if inference_framework == "onnx" else ".tflite"

        for path in models_dir.iterdir():
            name = path.name.lower()
            if "jarvis" in name and name.endswith(suffix):
                return str(path)

        raise FileNotFoundError(
            f"Could not find a 'hey jarvis' {suffix} model in {models_dir}.\n"
            "Make sure you've installed openwakeword and (if needed) run "
            "`openwakeword.utils.download_models()`."
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to automatically locate the 'hey jarvis' model. "
            "Please pass --model_path pointing to your my_jarvis model "
            "(.onnx or .tflite)."
        ) from exc


def load_asr_model(device: Optional[str] = None):
    """
    Load NVIDIA Parakeet TDT 0.6B v2 ASR model via NeMo.

    Requires: pip/uv install 'nemo_toolkit[asr]'
    """
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print(
            "ERROR: nemo_toolkit[asr] is not installed.\n"
            "Install it in your environment, e.g.:\n"
            "  uv add 'nemo_toolkit[asr]'\n"
            "or\n"
            "  pip install -U 'nemo_toolkit[asr]'\n",
            file=sys.stderr,
        )
        raise

    print("Loading ASR model nvidia/parakeet-tdt-0.6b-v2 ...", file=sys.stderr)
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )

    if device:
        try:
            asr_model.to(device)
        except Exception as exc:
            print(
                f"Warning: failed to move ASR model to device '{device}': {exc}. "
                "Falling back to default device.",
                file=sys.stderr,
            )

    return asr_model


def write_temp_wav(samples: np.ndarray, sample_rate: int) -> str:
    """
    Write mono int16 samples to a temporary WAV file and return the path.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    return tmp_path


def transcribe_with_parakeet(
    asr_model, samples: np.ndarray, sample_rate: int
) -> str:
    """
    Run Parakeet TDT 0.6B v2 on the given audio samples and return the text.
    """
    wav_path = write_temp_wav(samples, sample_rate)
    try:
        outputs = asr_model.transcribe([wav_path])
        if not outputs:
            return ""

        out0 = outputs[0]

        # NeMo typically returns an object with a `.text` attribute
        if hasattr(out0, "text"):
            return out0.text

        # Some NeMo utilities can return dicts
        if isinstance(out0, dict):
            return out0.get("text") or out0.get("pred_text", "")

        # Fallback to string
        return str(out0)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def compute_rms(audio: np.ndarray) -> float:
    """
    Root-mean-square amplitude for 16-bit PCM.
    """
    if audio.size == 0:
        return 0.0
    # Use float32 to avoid overflow
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "OpenWakeWord 'hey jarvis' listener + Parakeet TDT 0.6B v2 ASR.\n"
            "Listens for the 'hey jarvis' wakeword, then records an utterance\n"
            "until end-of-utterance silence and sends it to NVIDIA Parakeet\n"
            "for transcription. ASR is only called once per wake spike "
            "(first time score crosses wake_threshold)."
        )
    )

    parser.add_argument(
        "--chunk_size",
        help="Audio chunk size in samples (multiples of 80ms recommended).",
        type=int,
        default=1280,
    )
    parser.add_argument(
        "--model_path",
        help=(
            "Path to a specific openWakeWord model file (.onnx or .tflite). "
            "If omitted, the built-in 'hey jarvis' model will be used."
        ),
        type=str,
        default="",
    )
    parser.add_argument(
        "--inference_framework",
        help="Inference framework to use for openWakeWord ('onnx' or 'tflite').",
        type=str,
        choices=["onnx", "tflite"],
        default="onnx",
    )
    parser.add_argument(
        "--wake_threshold",
        help="Score threshold for triggering the wakeword spike (default 0.95).",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--silence_threshold",
        help=(
            "RMS amplitude below which a chunk is considered 'silence' for "
            "end-of-utterance detection (16-bit scale, default 500)."
        ),
        type=float,
        default=500.0,
    )
    parser.add_argument(
        "--silence_duration",
        help=(
            "How many seconds of continuous silence to treat as end-of-utterance "
            "(default 1.2s)."
        ),
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--max_utterance",
        help="Maximum utterance length in seconds before forcing end-of-utterance.",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--asr_device",
        help="Device for Parakeet ASR model (e.g. 'cuda' or 'cpu'). Default: NeMo default.",
        type=str,
        default="",
    )

    args = parser.parse_args(argv)

    # --- Set up microphone / PyAudio -------------------------------------------------
    chunk = args.chunk_size

    audio = pyaudio.PyAudio()
    mic_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=chunk,
    )

    # --- Load wakeword model ('hey jarvis' only) -------------------------------------
    if args.model_path:
        wakeword_models = [args.model_path]
        print(f"Using custom wakeword model: {args.model_path}")
    else:
        jarvis_path = find_hey_jarvis_model(args.inference_framework)
        wakeword_models = [jarvis_path]
        print(f"Using built-in 'hey jarvis' model: {jarvis_path}")

    oww_model = Model(
        wakeword_models=wakeword_models,
        inference_framework=args.inference_framework,
    )

    if not oww_model.models:
        print("ERROR: No wakeword models loaded.", file=sys.stderr)
        sys.exit(1)

    wake_model_name: Optional[str] = None  # will grab from prediction_buffer

    # --- Load ASR model --------------------------------------------------------------
    asr_model = load_asr_model(device=args.asr_device)

    # --- Utterance / end-of-utterance parameters ------------------------------------
    silence_chunks_required = max(
        1, int(args.silence_duration * RATE / chunk)
    )
    max_utterance_chunks = int(args.max_utterance * RATE / chunk)

    # --- Wake spike / debounce state -------------------------------------------------
    # We trigger only on the FIRST upward crossing of wake_threshold (spike),
    # then we wait for scores to fall below a reset threshold before allowing
    # another spike to trigger.
    prev_score: float = 0.0
    wake_armed: bool = True
    need_reset_after_command: bool = False
    reset_threshold: float = 0.3  # internal: score must fall below this to re-arm

    print("\n" + "#" * 80)
    print("Listening for wakeword: 'hey jarvis'")
    print(
        "After the FIRST spike above wake_threshold, your command is recorded "
        "and transcribed once. Pause to end the utterance."
    )
    print("#" * 80 + "\n")

    recording_command = False
    command_audio_chunks: List[np.ndarray] = []
    silence_run = 0
    utterance_chunks = 0

    try:
        while True:
            # Read from mic
            raw_bytes = mic_stream.read(chunk, exception_on_overflow=False)
            frame = np.frombuffer(raw_bytes, dtype=np.int16)

            if not recording_command:
                # Wakeword detection mode
                _ = oww_model.predict(frame)

                # Lazily determine the wake model name from prediction buffer
                if wake_model_name is None:
                    if not oww_model.prediction_buffer:
                        continue
                    wake_model_name = next(iter(oww_model.prediction_buffer.keys()))
                    print(f"Wakeword model name: {wake_model_name}")

                scores = list(oww_model.prediction_buffer[wake_model_name])
                current_score = float(scores[-1])

                # If we've just finished a command, wait for the model to "cool down"
                # below reset_threshold before we allow another spike to trigger.
                if need_reset_after_command:
                    if current_score < reset_threshold:
                        need_reset_after_command = False
                        wake_armed = True
                    prev_score = current_score
                    continue

                # Only trigger on the *first upward crossing* above wake_threshold
                if (
                    wake_armed
                    and prev_score < args.wake_threshold
                    and current_score >= args.wake_threshold
                ):
                    print(
                        f"[Wakeword] Spike detected for 'hey jarvis' "
                        f"(score={current_score:.3f}) - listening for command..."
                    )
                    recording_command = True
                    command_audio_chunks = []
                    silence_run = 0
                    utterance_chunks = 0

                    # Disarm further spikes until this command cycle completes
                    wake_armed = False

                prev_score = current_score

            else:
                # Command recording mode
                command_audio_chunks.append(frame)
                utterance_chunks += 1

                rms = compute_rms(frame)
                if rms < args.silence_threshold:
                    silence_run += 1
                else:
                    silence_run = 0

                end_by_silence = silence_run >= silence_chunks_required
                end_by_length = utterance_chunks >= max_utterance_chunks

                if end_by_silence or end_by_length:
                    # Concatenate all recorded chunks
                    if command_audio_chunks:
                        command_audio = np.concatenate(command_audio_chunks)
                    else:
                        command_audio = np.array([], dtype=np.int16)

                    if command_audio.size == 0:
                        print("[ASR] No audio captured for command, skipping.")
                    else:
                        print("[ASR] End of utterance detected, transcribing...")
                        text = transcribe_with_parakeet(
                            asr_model, command_audio, RATE
                        )
                        print(f"[ASR] Transcript: {text}")

                    # Reset to listening for wakeword again, but require
                    # the model's score to fall below reset_threshold before
                    # we treat any new high values as a new spike.
                    recording_command = False
                    command_audio_chunks = []
                    silence_run = 0
                    utterance_chunks = 0

                    need_reset_after_command = True
                    wake_armed = False
                    prev_score = 0.0

                    print("\nListening again for 'hey jarvis'...\n")

    except KeyboardInterrupt:
        print("\nExiting on Ctrl+C...")
    finally:
        try:
            mic_stream.stop_stream()
            mic_stream.close()
        except Exception:
            pass
        audio.terminate()


if __name__ == "__main__":
    main()
