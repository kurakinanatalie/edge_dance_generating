def install_edge_dependencies():
    import subprocess, sys

    pkgs = [
        "p_tqdm",
        "tqdm",
        "soundfile",
        "librosa",
        "einops",
        "matplotlib",
        "transformers",
    ]

    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
    print("[deps] Installed EDGE dependencies")