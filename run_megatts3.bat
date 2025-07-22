@echo off
echo Starting MegaTTS3-WaveVAE...
echo.

REM Activate conda environment
call conda activate megatts3-env

REM Check if activation was successful
if errorlevel 1 (
    echo Error: Could not activate conda environment 'megatts3-env'
    echo Please make sure the environment exists and conda is installed.
    pause
    exit /b 1
)

echo Environment activated successfully!
echo.

REM Change to the script directory (where the batch file is located)
cd /d "%~dp0"

REM Set PYTHONPATH to current directory
set PYTHONPATH=%CD%;%PYTHONPATH%
echo PYTHONPATH set to: %CD%
echo.

REM Run the MegaTTS3 Gradio interface
echo Starting MegaTTS3-WaveVAE Web Interface...
echo Open your browser and go to: http://localhost:7929
echo.
echo Press Ctrl+C to stop the server when done.
echo.

python tts/megatts3_gradio.py

REM Pause to see any error messages
if errorlevel 1 (
    echo.
    echo An error occurred while running the application.
    pause
)

echo.
echo MegaTTS3-WaveVAE has been stopped.
pause