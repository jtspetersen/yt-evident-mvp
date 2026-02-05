@echo off
REM Evident MVP Runner - Windows Batch Script
REM Usage: runvid [--infile <path>] [--channel <name>] [--review]

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found at .venv\
    echo Please create it with: python -m venv .venv
    exit /b 1
)

REM Activate virtual environment and run the app
call .venv\Scripts\activate.bat && python -m app.main %*
