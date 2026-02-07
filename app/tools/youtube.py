# app/tools/youtube.py
"""
YouTube transcript fetching with local Whisper fallback.

Fetches existing YouTube captions when available, falls back to
downloading audio and transcribing locally with faster-whisper.
"""
import glob
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


def _fmt_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS for normalize_transcript compatibility."""
    s = int(seconds)
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def extract_video_id(url_or_id: str) -> str:
    """Extract 11-char YouTube video ID from various URL formats.

    Supports: youtube.com/watch?v=, youtu.be/, shorts/, embed/, bare ID.
    Raises ValueError if no valid ID found.
    """
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s

    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(1)

    raise ValueError(f"Could not extract a YouTube video ID from: {s}")


def fetch_video_metadata(video_id: str) -> dict:
    """Use yt-dlp to extract video metadata without downloading.

    Returns dict with: title, channel, url, duration, upload_date.
    Returns partial dict on failure (url always present).
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    result = {
        "title": None,
        "channel": None,
        "url": url,
        "duration": None,
        "upload_date": None,
    }
    try:
        import yt_dlp
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            result["title"] = info.get("title")
            result["channel"] = info.get("channel") or info.get("uploader")
            result["duration"] = info.get("duration")
            result["upload_date"] = info.get("upload_date")
    except Exception:
        pass  # Return partial result with at least the URL
    return result


def _fetch_captions(video_id: str, languages=None) -> str | None:
    """Try to get YouTube captions via youtube-transcript-api.

    Prefers manually created > auto-generated, English variants.
    Returns timestamped text (M:SS format) joined by newlines, or None.
    """
    languages = languages or ["en", "en-US", "en-GB"]
    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            TranscriptsDisabled,
            NoTranscriptFound,
        )

        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # Sort: prefer manually created over auto-generated,
        # and prefer matching language codes
        manual = []
        generated = []
        for t in transcript_list:
            if t.language_code in languages:
                if t.is_generated:
                    generated.append(t)
                else:
                    manual.append(t)

        # Pick best: manual first, then generated
        chosen = (manual or generated or [None])[0]
        if chosen is None:
            return None

        fetched = chosen.fetch()
        lines = []
        for snippet in fetched.snippets:
            ts = _fmt_timestamp(snippet.start)
            lines.append(f"{ts} {snippet.text}")
        return "\n".join(lines)

    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None


def _find_ffmpeg() -> str | None:
    """Find ffmpeg binary, searching PATH and common install locations.

    Returns the directory containing ffmpeg/ffprobe, or None.
    """
    # Check PATH first
    if shutil.which("ffmpeg"):
        return None  # yt-dlp will find it via PATH

    search_dirs: list[str] = []

    if sys.platform == "win32":
        # winget installs to AppData\Local\Microsoft\WinGet\Packages\
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        if local_appdata:
            winget_pkgs = os.path.join(local_appdata, "Microsoft", "WinGet", "Packages")
            # Glob for Gyan.FFmpeg_* packages
            ffmpeg_globs = glob.glob(
                os.path.join(winget_pkgs, "Gyan.FFmpeg*", "ffmpeg-*", "bin")
            )
            search_dirs.extend(ffmpeg_globs)

        # Common Windows install paths
        for pf in [os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", "")]:
            if pf:
                search_dirs.append(os.path.join(pf, "ffmpeg", "bin"))

        # Chocolatey
        choco = os.environ.get("ChocolateyInstall", r"C:\ProgramData\chocolatey")
        search_dirs.append(os.path.join(choco, "bin"))

        # Scoop
        userprofile = os.environ.get("USERPROFILE", "")
        if userprofile:
            search_dirs.append(os.path.join(userprofile, "scoop", "shims"))

    else:
        # Linux / macOS common paths
        search_dirs.extend(["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"])

    for d in search_dirs:
        ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        if os.path.isfile(os.path.join(d, ffmpeg_name)):
            return d

    return None


def _download_and_transcribe(url: str, progress_callback=None) -> str:
    """Download audio via yt-dlp, transcribe with faster-whisper.

    Uses tempfile.TemporaryDirectory for automatic cleanup.
    Raises RuntimeError on failure.
    """
    def _cb(data):
        if progress_callback:
            progress_callback(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Download audio
        _cb({"status": "downloading_audio", "detail": "Downloading audio from YouTube..."})
        import yt_dlp
        outtmpl = str(tmp_path / "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "noplaylist": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }
        ffmpeg_dir = _find_ffmpeg()
        if ffmpeg_dir:
            ydl_opts["ffmpeg_location"] = ffmpeg_dir
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        audio_path = tmp_path / "audio.mp3"
        if not audio_path.exists():
            raise RuntimeError(
                "Audio download failed (audio.mp3 not found). Is ffmpeg installed?"
            )

        # Transcribe with Whisper
        _cb({"status": "transcribing", "detail": "Transcribing with Whisper (this may take a few minutes)..."})
        from faster_whisper import WhisperModel
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(audio_path), beam_size=5)
        lines = []
        for seg in segments:
            ts = _fmt_timestamp(seg.start)
            lines.append(f"{ts} {seg.text.strip()}")
        text = "\n".join(lines).strip()

        if not text:
            raise RuntimeError("Whisper produced an empty transcript.")

        return text


def fetch_youtube_transcript(url: str, progress_callback=None) -> dict:
    """Fetch transcript from a YouTube URL.

    Tries YouTube captions first, falls back to local Whisper transcription.

    Returns:
        {
            "raw_text": str,
            "source": "captions" | "whisper",
            "video_id": str,
            "metadata": {"title", "channel", "url", "duration", "upload_date"},
        }

    progress_callback receives dicts with "status" key:
        extracting_id, fetching_metadata, trying_captions, captions_found,
        no_captions, downloading_audio, transcribing, done
    """
    def _cb(data):
        if progress_callback:
            progress_callback(data)

    # Extract video ID
    _cb({"status": "extracting_id", "detail": "Parsing YouTube URL..."})
    video_id = extract_video_id(url)

    # Fetch metadata (title, channel, etc.)
    _cb({"status": "fetching_metadata", "detail": "Fetching video metadata..."})
    metadata = fetch_video_metadata(video_id)

    # Try YouTube captions first
    _cb({"status": "trying_captions", "detail": "Looking for existing captions..."})
    caption_text = _fetch_captions(video_id)

    if caption_text:
        _cb({"status": "captions_found", "detail": "Captions found!"})
        _cb({"status": "done", "source": "captions", "detail": f"Transcript ready ({len(caption_text)} chars)"})
        return {
            "raw_text": caption_text,
            "source": "captions",
            "video_id": video_id,
            "metadata": metadata,
        }

    # Fallback: download audio and transcribe with Whisper
    _cb({"status": "no_captions", "detail": "No captions available, falling back to Whisper..."})
    whisper_text = _download_and_transcribe(
        metadata["url"],
        progress_callback=progress_callback,
    )

    _cb({"status": "done", "source": "whisper", "detail": f"Transcript ready ({len(whisper_text)} chars)"})
    return {
        "raw_text": whisper_text,
        "source": "whisper",
        "video_id": video_id,
        "metadata": metadata,
    }
