import tkinter as tk
import cv2
from tkinter import filedialog
from PIL import Image, ImageTk
from preprocessing.face_detection import detect_face


class EmpathyApp:
    # Singleton instance (only 1 window interface)
    _instance = None

    mat_dim = (640, 480)
    photo_ext = ("Photo files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp;*.tiff")
    vid_ext = ("Video files", "*.mp4;*.mkv;*.avi;*.mov;*.flv")

    button_style = {
        "padx": 20,
        "pady": 10,
        "bg": "#3498db",
        "fg": "#ffffff",
        "font": "Roboto",
        "relief": tk.FLAT
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmpathyApp, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.rootwnd = None
        self.media_path = None
        self.mediabox = None
        self.cap_obj = None
        self.uploader = None
        self.realtime = False
        self.statsbox = None

        self.fast_validation_photo = self.photo_ext[1]

        self.__strap()

    def on(self):
        assert self.rootwnd is not None
        self.rootwnd.mainloop()

    def __strap(self):

        # Window properties
        self.rootwnd = tk.Tk()
        self.rootwnd.title("Empathy")
        self.rootwnd.configure(bg="#1d2a36")
        self.__window_center()
        self.__build_root_grid_unif()

        # Graphic design part of the UI
        self.rootwnd.iconphoto(False, tk.PhotoImage(file="gui/graphics/logo_notransp.png"))
        img = Image.open("gui/graphics/eyelogo.png")
        img.thumbnail((220, 220), Image.Resampling.LANCZOS)
        logo = ImageTk.PhotoImage(img, Image.Resampling.LANCZOS)
        title_label = tk.Label(self.rootwnd, image=logo,
                               font=("Roboto", 20), bg="#2c3e50", fg="#ecf0f1")
        title_label.image = logo

        # Buttons
        self.uploader = tk.Button(self.rootwnd, text="Upload Media", command=self.__upload, **self.button_style)
        self.camera = tk.Button(self.rootwnd, text="Toggle Camera", command=self.camera_on, **self.button_style)
        self.delete_cap = tk.Button(self.rootwnd, text="Stop Capture", command=self.stop_capture, **self.button_style)

        # Media display
        self.mediabox = tk.Label(self.rootwnd, bg="#34495e")

        # Emotion Detection Statistics display
        stats_title = tk.Label(self.rootwnd, text="Emotional Statistics",
                               font=("Roboto", 16), bg="#2c3e50", fg="#ecf0f1")

        self.statsbox = tk.Text(self.rootwnd, height=10, width=30, bg="#34495e", fg="#ecf0f1",
                                font=("Roboto", 12), relief=tk.FLAT)
        self.statsbox.config(state=tk.DISABLED)

        # Grid positions
        title_label.grid(
            row=0, column=0, columnspan=9, pady=10, sticky="nsew"
        )
        stats_title.grid(
            row=0, column=9, columnspan=3, pady=10, sticky="nsew"
        )
        self.mediabox.grid(
            row=2, column=0, columnspan=9, rowspan=9, padx=10, pady=10, sticky="nsew"
        )
        self.uploader.grid(
            row=1, column=3, pady=5, sticky="nsew"
        )
        self.camera.grid(
            row=1, column=4, pady=5, sticky="nsew"
        )
        self.delete_cap.grid(
            row=1, column=5, pady=5, sticky="nsew"
        )
        self.statsbox.grid(
            row=1, column=9, rowspan=9, columnspan=3, padx=10, pady=10, sticky="nsew"
        )

    def __build_root_grid_unif(self):
        # Configure grid columns and rows to be responsive
        for i in range(12):
            self.rootwnd.columnconfigure(i, weight=1)
        for j in range(10):
            self.rootwnd.rowconfigure(j, weight=1)

    def __window_center(self):
        screen_width = self.rootwnd.winfo_screenwidth()
        screen_height = self.rootwnd.winfo_screenheight()

        width_set = screen_width // 2
        height_set = screen_height // 2 + 150

        offs_x = (screen_width - width_set) // 2
        offs_y = (screen_height - height_set) // 2

        self.rootwnd.geometry(f'{width_set}x{height_set}+{offs_x}+{offs_y}')

    def __upload(self):
        media_uploaded = filedialog.askopenfile(filetypes=[self.photo_ext, self.vid_ext])

        if media_uploaded is not None:
            type_check = media_uploaded.name.split(".")[-1]

            if f'.{type_check}' in self.fast_validation_photo:
                self.cap_obj = cv2.imread(media_uploaded.name, 1)
                self.realtime = False
            else:
                self.cap_obj = cv2.VideoCapture(media_uploaded.name)
                self.realtime = True

        self.disp_media()

    def camera_on(self):
        self.cap_obj = cv2.VideoCapture(0)

        if self.cap_obj.isOpened():
            self.realtime = True
            self.disp_media()

    def stop_capture(self):
        if self.cap_obj is not None and self.realtime:
            self.cap_obj.release()
            self.cap_obj = None
            self.realtime = False
            self.mediabox.configure(image='')

    def disp_media(self):
        if self.cap_obj is not None:
            if self.realtime:
                flag, frame = self.cap_obj.read()
            else:
                flag, frame = True, self.cap_obj

            if flag:
                frame, processed_faces = detect_face(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL image and then to ImageTk format (compatibility)
                img_pil = Image.fromarray(frame)
                img_pil.thumbnail(self.mat_dim, Image.Resampling.LANCZOS)
                img_adapted = ImageTk.PhotoImage(img_pil)

                self.mediabox.image = img_adapted
                self.mediabox.configure(image=img_adapted)

                # Continue updating if video (realtime)
                if self.realtime:
                    self.rootwnd.after(15, self.disp_media)
            else:
                if self.realtime:
                    self.cap_obj.release()
                self.cap_obj = None
                self.mediabox.configure(image='')
                self.uploader.grid(row=10, column=1)
                self.camera.grid(row=11, column=1)
