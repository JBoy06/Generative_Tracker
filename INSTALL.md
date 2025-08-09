Generative_Tracker Installation Instructions
===========================================

1. **Clone the repository:**
   ```
   git clone https://github.com/JBoy06/Generative_Tracker.git
   cd Generative_Tracker
   ```

2. **Install dependencies (Ubuntu):**
   ```
   sudo apt-get update
   sudo apt-get install parallel ffmpeg bc python3 python3-pip
   pip3 install -r requirements.txt
   ```

3. **(Optional) Set up Python virtual environment:**
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Make scripts executable:**
   ```
   chmod +x generative_tracker/*.sh
   ```

5. **Run a sample script:**
   ```
   ./generative_tracker/check_video.sh sample.mp4
   ./generative_tracker/run_parallel.sh sample.mp4
   ```

See README.md for detailed usage.
