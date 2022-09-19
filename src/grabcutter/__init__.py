from grabcutter.grabcutter import PyGrabCutter

def main():
    import sys
    print(sys.argv)
    fname = None
    if len(sys.argv) == 2:
        fname = sys.argv[1]
    pgc = PyGrabCutter(fname)
