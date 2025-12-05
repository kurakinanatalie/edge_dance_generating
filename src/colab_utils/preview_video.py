from pathlib import Path
from base64 import b64encode
from IPython.display import display, Video, HTML


def show_last_video(renders_dir: Path, width: int = 720):
    """
    Find the most recently modified mp4/webm in renders_dir and display it in the notebook.
    Also prints a small list of recent files.
    """
    renders_dir = Path(renders_dir)
    videos = sorted(
        [p for p in renders_dir.rglob("*") if p.suffix.lower() in (".mp4", ".webm")],
        key=lambda p: p.stat().st_mtime,
    )
    if not videos:
        print("No video found in", renders_dir)
        return

    latest = videos[-1]
    print("Showing last file:", latest)

    # Try direct embedding via IPython.display.Video
    try:
        display(Video(str(latest), embed=True, width=width))
    except Exception as e:
        print("Video() failed, falling back to base64:", e)
        data = latest.read_bytes()
        b64 = b64encode(data).decode("ascii")
        mime = "video/mp4" if latest.suffix.lower() == ".mp4" else "video/webm"
        html = f"""
        <video width="{width}" controls preload="metadata" style="outline:1px solid #333">
          <source src="data:{mime};base64,{b64}" type="{mime}">
          Your browser does not support the video tag.
        </video>
        """
        display(HTML(html))

    print("\nRecent files:")
    for p in videos[-10:]:
        print(" -", p)