import sys
import os
import os.path
from cx_Freeze import setup, Executable

def path(path:list):
    return os.path.join(*path)

PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
# os.environ['TCL_LIBRARY'] = rf'`{PYTHON_INSTALL_DIR}\Library\lib\tcl8.6'
# os.environ['TK_LIBRARY'] = rf'{PYTHON_INSTALL_DIR}\Library\lib\tcl8.6'
# print(PYTHON_INSTALL_DIR)
os.environ['TCL_LIBRARY'] = rf'`{PYTHON_INSTALL_DIR}\Library\lib\tcl8.6'
os.environ['TK_LIBRARY'] = rf'{PYTHON_INSTALL_DIR}\Library\lib\tk8.6'

base = None
if sys.platform == 'win32':  # for GUI app
    base = 'Win32GUI'

exe = Executable(script=path(['..','src','grabcutter','main.py']),
                 base=base)

opts = {
    'build': {
        'build_exe': 'dist',
    },
    'build_exe': {
        # 'optimize': '2',
        'optimize': '1',  # workaround for bug https://github.com/numpy/numpy/issues/13248
        'packages': [
            'numpy',
            'numpy.lib.format',
            ],
        'include_files':[
            os.path.join(PYTHON_INSTALL_DIR, 'Library', 'bin', 'tk86t.dll'),
            os.path.join(PYTHON_INSTALL_DIR, 'Library', 'bin', 'tcl86t.dll')
            # "tcl86t.dll",
            # "tk86t.dll",
         ],
    },
    'install': {
        'install_exe': 'inst',
    },
    'install_exe': {
        'install_dir':
            os.environ['HOMEPATH'] + r'\AppData\Local\grabcutter\bin',
    },
}


setup(name='grabcutter',
      version='0.1',
      options=opts,
      executables=[exe])
