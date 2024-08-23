import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk


class GInterface:

    # singleton instance (only 1 window interface)
    _instance = None

    mat_dim = (640, 360)
    photo_ext = ("Photo files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp;*.tiff")
    vid_ext = ("Video files", "*.mp4;*.mkv;*.avi;*.mov;*.flv")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GInterface, cls).__new__(cls)
        return cls._instance

    def __init__(self):

        self.rootwnd =     None
        self.media_path =   None
        self.mediabox =     None
        self.cap_obj =      None
        self.uploader =     None

        self.realtime =     False

        self.fast_validation_photo = self.photo_ext[1]
        self.fast_validation_video = self.vid_ext[1]

        self.__strap()

    def on(self):
        assert self.rootwnd is not None
        self.rootwnd.mainloop()

    def __strap(self):

        self.rootwnd = tk.Tk()
        self.rootwnd.title("Empathy")
        self.rootwnd.configure(bg="#606060")
        self.__window_center()
        self.__build_root_grid_unif()

        self.uploader = (tk.Button(self.rootwnd,
                                   padx="20",
                                   pady="3",
                                   bg="#5dbea3", fg="#000000",
                                   relief=tk.FLAT,
                                   text="Upload Media",
                                   font="Roboto",
                                   command=self.__upload)
                         )

        self.uploader.grid(row=10, column=1)

        self.mediabox = tk.Label(self.rootwnd, bg="#606060")
        self.mediabox.grid(row=3, column=1)

    def __build_root_grid_unif(self):

        for i in range(11):
            self.rootwnd.columnconfigure(i, weight=1)
            for j in range(10):
                self.rootwnd.rowconfigure(i, weight=1)

    def __window_center(self):

        screen_width = self.rootwnd.winfo_screenwidth()
        screen_height = self.rootwnd.winfo_screenheight()

        width_set = (screen_width + 50) // 2
        height_set = screen_height // 2

        offs_x = width_set // 2
        offs_y = height_set // 2

        self.rootwnd.geometry(f'{width_set}x{height_set}+{offs_x}+{offs_y}')

    def __upload(self):

        media_uploaded = (
            filedialog
            .askopenfile(filetypes=[self.photo_ext, self.vid_ext])
        )

        if media_uploaded is not None:
            type_check = media_uploaded.name.split(".")[-1]

            if f'.{type_check}' in self.fast_validation_photo:
                self.cap_obj = cv2.imread(media_uploaded.name, 1)
                self.realtime = False
            else:
                self.cap_obj = cv2.VideoCapture(media_uploaded.name)
                self.realtime = True

        self.disp_media()


    def disp_media(self):
        """
        Logic to display uploaded media in the (4,1) position on the window grid
        :return: void
        """

        if self.cap_obj is not None:

            if self.realtime:
                flag, frame = self.cap_obj.read()
            else:
                flag, frame = True, self.cap_obj

            if flag:
                self.uploader.grid(row=4, column=1)

                # set fixed resolution
                frame = cv2.resize(frame, self.mat_dim)

                # convert from cv2 standard BGR to fit to tkinter framework
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # tkinter can use pillow framework
                img_pil = Image.fromarray(frame)
                img_adapted = ImageTk.PhotoImage(img_pil)

                # set image and avoid garbage collection (First line)
                self.mediabox.image = img_adapted
                self.mediabox.configure(image=img_adapted)

                if self.realtime:
                    self.rootwnd.after(15, self.disp_media)
            else:
                if self.realtime:
                    self.cap_obj.release()

                self.cap_obj = None
                self.uploader.grid(row=10, column=1)
        else:
            self.mediabox = tk.Label(self.rootwnd)