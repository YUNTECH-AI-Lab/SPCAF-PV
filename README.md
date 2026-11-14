# SPCAF-PV
 A Synthetic Physics-based Cloud-Aware Framework for Residential PV Fleet Generation Dataset
## Installation and Local Use on Windows

SynPV is a Python application with a browser-based interface. It runs locally on your computer and opens in your web browser.

### 1. Install Python

Install **Python 3.11 (64-bit)**. During installation, select **Add Python to PATH**.

### 2. Download and Extract the Project

Download the project ZIP file and extract it. Open the `SynPV` folder containing `pyproject.toml` and `requirements.txt`.

To open Command Prompt in this folder, type `cmd` in the File Explorer address bar and press **Enter**.

### 3. Create and Activate a Virtual Environment

Run the following commands in Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

After activation, `(.venv)` should appear at the beginning of the command prompt.

### 4. Install the Application

```bat
python -m pip install --upgrade pip
python -m pip install -e .
```

The final dot in the second command is required: it tells Python to install the project from the current folder.

An internet connection is required to download the dependencies.

### 5. Launch the Application

```bat
python -m synpv app
```

The application should open automatically in your default browser. If it does not, open:

http://localhost:8050

Keep the Command Prompt window open while using the application. To stop the application, press **Ctrl+C**.

### 6. Generate a Dataset

In the application:

1. Select a location or enter its coordinates.
2. Choose the simulation dates.
3. Set the number of houses and other simulation parameters.
4. Click **Generate dataset**.

For your first run, use a short simulation period and a small number of houses.

The application requires internet access to retrieve NASA POWER weather data and the CEC equipment databases when they are not already cached.

Generated datasets are saved in the `synpv_runs` folder.

### Running the Application Again

Installation is only required once. For subsequent sessions, open Command Prompt in the project folder and run:

```bat
.venv\Scripts\activate
python -m synpv app
```

### Troubleshooting

- **`py` is not recognized:** Check that Python is installed and available on PATH. Reopen Command Prompt after installation.
- **`No module named synpv`:** Activate the virtual environment and run `python -m pip install -e .` from the folder containing `pyproject.toml`.
- **The browser does not open:** Manually visit http://localhost:8050.
- **Port 8050 is already in use:** Run `python -m synpv app --port 8051`, then open http://localhost:8051.
