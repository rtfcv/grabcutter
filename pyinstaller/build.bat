@echo off
WHERE pyinstaller.exe || (
    @echo PYINSTALLER NOT FOUND
    exit
)
@echo on
python.exe -mvenv tmpEnv
.\tmpEnv\Scripts\pip.exe install -U -r .\requirements.txt

pyinstaller.exe ^
  -p .\tmpEnv\Lib\site-packages ^
  --exclude-module setuptools ^
  --exclude-module pip ^
  ..\src\grabcutter\grabcutter.py

.\tmpEnv\Scripts\pip.exe freeze
RMDIR /S /Q tmpEnv
