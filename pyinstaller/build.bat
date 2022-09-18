@echo off
pyinstaller.exe > NUL
@echo on
python.exe -mvenv tmpEnv
.\tmpEnv\Scripts\pip.exe install -U -r .\requirements.txt
pyinstaller.exe -p .\tmpEnv\Lib\site-packages ..\src\grabcutter\grabcutter.py
RMDIR /S /Q tmpEnv
