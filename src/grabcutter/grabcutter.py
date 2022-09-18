import tkinter
from tkinter import ttk
from tkinter import filedialog
from typing import List, Union, Tuple, Any, Dict
from os import path
from numpy import array
# from numpy.linalg import norm
import numpy as np
from math import pi
import PIL.Image
import PIL.ImageTk
import json
import cv2

deg: float = pi / 180.0


def dummyHook(draggedObj: Union[Tuple[int], None]) -> None:
    _ = draggedObj


def toJson(fname: str, obj: Any) -> None:
    with open(fname, mode='w') as file:
        returnValue = json.dump(
            obj, file, ensure_ascii=False, indent=2, sort_keys=False)
    return returnValue


def fromJson(fname: str) -> Any:
    with open(fname, mode='r') as file:
        returnVaule = json.load(
            file,)
    return returnVaule


class EntryVariable:
    def getInt(self) -> int:
        ''' Get value of input field as int
        will fall back to float and cast it as int
        '''
        try:
            returnVal = int(self.stringVar.get())
        except ValueError as e:
            print(e)
            returnVal = int(self.getFloat())
        return returnVal

    def getFloat(self) -> float:
        ''' Get value of input field as float'''
        return float(self.stringVar.get())

    def set(self, value: str):
        return self.stringVar.set(value)

    def __init__(self, parent: ttk.Frame, *args, **kwargs):
        self.stringVar = tkinter.StringVar() # contains actual stringvar
        self.entry = ttk.Entry(parent, # contains actual entry object
                               *args,
                               textvariable=self.stringVar,
                               **kwargs)
        self.pack = self.entry.pack



class PyGrabCutter:
    def __init__(self, fname: Union[None, str]=None):
        self.imgPath = ''

        self.root = tkinter.Tk()
        self.root.title('hoge')

        self.style = ttk.Style()
        print(self.style.theme_names())
        # self.style.theme_use('classic')
        self.style.theme_use('clam')

        self.leftPane = ttk.Frame(self.root, padding=2)
        # self.centerPane = ttk.Frame(self.root, padding=2)
        self.rightPane = ttk.Frame(self.root, padding=2)

        self.frame1 = ttk.Frame(self.rightPane, padding=2)
        self.label1 = ttk.Label(self.frame1, text='Filename:')
        self.fname = tkinter.StringVar()
        self.fnameEntry = ttk.Entry(self.frame1, textvariable=self.fname,)
        self.fnameEntry.configure(state='readonly')
        self.button1 = ttk.Button(
            self.frame1,
            text='Select File',
            command=self.selectImage
        )
        self.SaveImgButton = ttk.Button(
            self.frame1,
            text='Save Image',
            command=self.saveImgOutput
        )
        self.ResizeToWinButton = ttk.Button(
            self.frame1,
            text='Resize to Window',
            command=lambda: self.updateSize(None)
        )

        self.t2 = tkinter.StringVar(value='hoge')
        self.frame2 = ttk.Frame(self.leftPane, padding=2)
        self.modeButtonFrame = ttk.Frame(self.frame2, padding=2)
        self.processButton = ttk.Button(
            self.modeButtonFrame,
            text='Refresh',
            command=self.grabAndCut
        )
        # self.img = None
        self.modeManager = ModeManager('modeman', self, self.modeButtonFrame)

        self.afterImageFrame = ttk.Frame(self.rightPane, padding=2)
        self.afterImage = ImgCanvas('after', self, self.afterImageFrame)

        self.beforeImageFrame = ttk.Frame(self.frame2, padding=2)
        self.beforeImage = ImgCanvas('after', self, self.beforeImageFrame)

        # self.geomEntries += self.centerYZ.geomEntries

        # pack stuffs
        self.leftPane.pack(side=tkinter.LEFT)
        self.rightPane.pack(side=tkinter.RIGHT)

        self.frame1.pack(anchor=tkinter.N)
        self.label1.pack( side=tkinter.LEFT)
        self.fnameEntry.pack( side=tkinter.LEFT)
        self.button1.pack(side=tkinter.LEFT)
        self.SaveImgButton.pack(side=tkinter.LEFT)
        self.ResizeToWinButton.pack(side=tkinter.LEFT) # this is broken for now

        self.modeButtonFrame.pack()
        self.modeManager.pack()
        self.processButton.pack(side=tkinter.LEFT)

        self.frame2.pack()
        self.beforeImageFrame.pack()
        self.beforeImage.pack()

        self.afterImageFrame.pack()
        self.afterImage.pack()

        self.dataInterface = DataSaver(self)

        # self.root.geometry("1000x600")

        if fname is not None:
            self.root.update_idletasks()
            self.root.update()
            self.openImage(fname)
        # self.root.state('zoomed')
        # self.updateSize(None)
        self.root.mainloop()

    def openImage(self, fname:str):
        self.afterImage.LoadImage(fname)
        self.afterImage.renderImg()
        self.beforeImage.LoadImage(fname)
        self.beforeImage.renderImg()
        # self.root.bind("<Configure>", self.updateSize)
        self.fname.set(fname)

        assert self.beforeImage.cvimg is not None
        self.origImg = self.beforeImage.cvimg.copy()
        self.rect_or_mask = 0

        # self.beforeImage.onDrag(lambda event: print(f'clicked at {event}'))

        self.prevPT: Union[None, np.ndarray] = None
        def resetPrevPt(event):
            self.prevPT=None
        self.beforeImage.onClick(resetPrevPt)
        self.beforeImage.onDrag(self.drawCB)

    def selectImage(self):
        fname: str = filedialog.askopenfilename(
            filetypes=[("Images", ('*.png', '*.jpg', '*.jpeg'))])
        if (type(fname) is str):
            if len(fname) > 0:
                self.openImage(fname)
        return

    def selectData(self):
        fname: str = filedialog.askopenfilename(
            filetypes=[("Saved Data", ('*.json'))])
        if (type(fname) is str):
            self.loadData(fname)
        return

    def loadData(self, fname: str):
        data = dict(fromJson(fname))

        return self.dataInterface.fromDict(data)

    def updateResult(self):
        return

    def updateSize(self, event):
        print(event)
        self.afterImage.renderImg()
        self.beforeImage.renderImg()

    def saveImgOutput(self):
        assert self.afterImage.cvmask is not None
        fname: str = filedialog.asksaveasfilename(
            filetypes=[("PNG File", ('*.png'))])

        if (type(fname) is str) and (len(fname) > 0):
            rgba = cv2.cvtColor(self.origImg, cv2.COLOR_RGB2RGBA)

            # where mask is 3 or 1 is 255 else 0
            rgba[:, :, 3] = np.where(
                    (self.afterImage.cvmask==3) + (self.afterImage.cvmask==1),255,0).astype('uint8')

            # np.savetxt('mask.csv', np.array(rgba[:, :, 3], dtype=int))

            cv2.imwrite(f'{fname}_mask.png', rgba[:, :, 3])

            if fname[-4:] != '.png':
                fname = f'{fname}.png'
            cv2.imwrite(f'{fname}', rgba)
            print(" Result saved as image \n")

    def drawCB(self, event):
        '''
        '''

        # disable Drag Handler for a bit
        self.beforeImage.canvas.unbind_all('<B1-Motion>')

        view_coords = np.array([event.x,event.y], dtype=int)
        __prevPT = self.prevPT
        coords = np.array(view_coords/self.beforeImage.scale, dtype=int)
        self.prevPT = coords
        color = np.array([255,255,255], dtype=int)

        mode = int(self.modeManager.get())  # 0 when bg, 1 when fg marking mode

        print(f'mark mode: {"bg" if mode==0 else "fg" }')
        col = (color*mode).tolist()  # should basically be [255,255,255] or [0,0,0]
        linewidth = 3

        cv2.circle(
            self.beforeImage.cvimg,
            tuple(coords),
            linewidth,
            col,
            -1)

        cv2.circle(
            self.afterImage.cvmask,
            tuple(coords),
            linewidth,
            int(1*mode),
            -1)

        if __prevPT is not None:
            cv2.line(
                self.beforeImage.cvimg,
                tuple(__prevPT),
                tuple(coords),
                col,
                linewidth)
            cv2.line(
                self.afterImage.cvmask,
                tuple(__prevPT),
                tuple(coords),
                int(1*mode),
                linewidth)

        self.beforeImage.renderImg()

        # reenable Drag Handler
        self.beforeImage.onDrag(self.drawCB)

    def grabAndCut(self):
        assert self.origImg is not None
        assert self.afterImage.cvmask is not None
        shp=self.afterImage.cvmask.shape
        rect = [0,0,shp[1]-1,shp[0]-1]
        # rect = (15, 137, 986, 338) # need to set rect

        print(self.origImg.shape)
        print(self.afterImage.cvmask.shape)
        if (self.rect_or_mask == 0):         # grabcut with rect
            bgdmodel = np.zeros((1,65),np.float64)
            fgdmodel = np.zeros((1,65),np.float64)

            cv2.grabCut(
                    self.origImg,
                    self.afterImage.cvmask,
                    rect,
                    bgdmodel,
                    fgdmodel,
                    1,
                    cv2.GC_INIT_WITH_RECT
                    )
            self.rect_or_mask = 1
        elif self.rect_or_mask == 1:         # grabcut with mask
            bgdmodel = np.zeros((1,65),np.float64)
            fgdmodel = np.zeros((1,65),np.float64)

            cv2.grabCut(
                    self.origImg,
                    self.afterImage.cvmask,
                    rect,
                    bgdmodel,
                    fgdmodel,
                    1,
                    cv2.GC_INIT_WITH_MASK
                    )

        mask2 = np.where((self.afterImage.cvmask==1) + (self.afterImage.cvmask==3),255,0).astype('uint8')
        print(mask2)
        self.afterImage.cvimg = cv2.bitwise_and(self.origImg,self.origImg,mask=mask2)
        self.afterImage.renderImg()


class ImgCanvas:
    def onClick(self, callback):
        self.canvas.unbind_all('<Button-1>')
        self.canvas.bind('<Button-1>', callback)

    def onRelease(self, callback):
        self.canvas.bind('<Button-1>', callback)

    def onDrag(self, callback):
        self.canvas.unbind_all('<B1-Motion>')
        self.canvas.bind('<B1-Motion>', callback)
        

    def LoadImage(self, fname:str):
        self.cvimg = cv2.imread(fname)
        self.cvmask = np.zeros_like(self.cvimg[:,:,0], dtype = np.uint8)
        # wierd color
        self.imgPath = fname

    def InitWindowSize(self):
        assert self.cvimg is not None

    def ScaleImage(self):
        assert self.cvimg is not None
        assert self.cvmask is not None

        self.screengeom = array([
            self.root.root.winfo_screenwidth()*0.5,
            self.root.root.winfo_screenheight()
        ], dtype=int) * 0.8
        
        # self.screengeom = array([
        #     self.root.root.winfo_height()*0.45,
        #     self.root.root.winfo_width()*0.45
        #     ], dtype=int)
        # self.sceengeom = array([self.canvas.winfo_width(), self.canvas.winfo_height()], dtype=int)

        self.igeom = self.cvimg.shape[1::-1]
        print(self.igeom)
        newGeom = self.igeom

        # resize image
        # before this could really be introduced,
        # a way to translate coordninates according to the scale must also be put in place
        if newGeom[0] > self.screengeom[0]:
            newGeom = array(np.array(newGeom, dtype=float) * self.screengeom[0] / newGeom[0], dtype=int)
        if newGeom[1] > self.screengeom[1]:
            newGeom = array(np.array(newGeom, dtype=float) * self.screengeom[1] / newGeom[1], dtype=int)

        self.showGeom = newGeom
        self.scale = float(newGeom[0])/float(self.igeom[0])
        print(newGeom)

    def renderImg(self):
        assert self.cvimg is not None
        assert self.cvmask is not None

        tmpimg = PIL.Image.fromarray(self.cvimg[:,:,[2,1,0]])

        self.ScaleImage()

        self.img = PIL.ImageTk.PhotoImage(image=tmpimg.resize(tuple(self.showGeom)))
        self.canvas.configure(width=self.showGeom[0], height=self.showGeom[1])
        self.canvas.create_image(0, 0, image=self.img, anchor=tkinter.NW)

        self.canvas.unbind_all('<Motion>')
        self.canvas.unbind_all('<Enter>')

        self.canvas.bind('<Motion>', self.get_coordinates)
        self.canvas.bind('<Enter>',  self.get_coordinates)  # handle <Alt>+<Tab> switches between windows
        self.tag = self.canvas.create_text(10, 10, text='', anchor='nw')

        self.root.root.update()

    def get_coordinates(self, event):
        assert(type(self.canvas) is tkinter.Canvas)
        self.canvas.itemconfigure(
            self.tag, text='({x}, {y})'.format(x=event.x, y=event.y))

    def pack(self) -> None:
        return self.canvas.pack(side=tkinter.TOP)
        # return self.canvas.pack(fill="both", expand=True)

    def __init__(self, name: str, root: PyGrabCutter, parent: ttk.Frame):
        self.name = name
        self.root = root
        self.cvimg:Union[None,np.ndarray] = None
        self.cvmask:Union[None,np.ndarray] = None
        self.scale:Union[None,float] = None
        self.canvas = tkinter.Canvas(parent,)


class ModeManager:
    def pack(self):
        self.drawBgButton.pack(side=tkinter.LEFT)
        self.drawFgButton.pack(side=tkinter.LEFT)

    def get(self):
        return self.mode.get()

    def __defButtons(self):
        self.drawBgButton = ttk.Radiobutton(
                    self.parent,
                    text='Draw Background',
                    value=0,
                    variable=self.mode,
            )
        self.drawFgButton = ttk.Radiobutton(
                    self.parent,
                    text='Draw Foreground',
                    value=1,
                    variable=self.mode,
            )

    def __init__(self, name: str, root: PyGrabCutter, parent: ttk.Frame):
        self.name = name
        self.root = root
        self.parent = parent
        self.mode = tkinter.IntVar()

        self.__defButtons()


class DataSaver:
    def __extract__entries(
        self,
        component
    ) -> Dict[str, EntryVariable]:
        dict__ = component.__dict__
        return {
            key: dict__[key]
            for key in dict__
            if (type(dict__[key]) is EntryVariable)
        }

    def __extract__data(self, keys: Union[List[str], Tuple[str]]):
        return {
            topKey: {
                key: self.__dict__[topKey][key].getFloat()
                for key in self.__dict__[topKey]
            } for topKey in keys
        }

    def update(self):
        pass

    def toDict(self):
        data = {}
        data['Files'] = {'path': self.parent.imgPath}
        data.update(self.__extract__data(['wing', 'hstab', 'centerXY']))
        return data

    def fromDict(self, data: dict):
        for topkey in data:
            if topkey in self.__dict__.keys():
                lhs = self.__dict__[topkey]
                rhs = data[topkey]
                for key in rhs:
                    if key in lhs.keys():
                        lhs[key].set(str(rhs[key]))
        return

    def __init__(self, parent: PyGrabCutter) -> None:
        self.parent = parent
        self.update()

if __name__ == '__main__':
    import sys
    print(sys.argv)
    fname = None
    if len(sys.argv) == 2:
        fname = sys.argv[1]
    pgc = PyGrabCutter(fname)
