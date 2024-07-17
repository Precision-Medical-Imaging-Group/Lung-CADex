import subprocess

# Install necessary Python packages
subprocess.run(["pip", "install", "numpy", "opencv-python", "librosa", "matplotlib"], check=True)

# Install the Segment Anything library directly from the GitHub repository
subprocess.run(["pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"], check=True)

print("Installation completed successfully!")
