import sys
import tkinter
import tkinter.filedialog
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *
import os.path
from functools import partial
from utils.hparams import hparams
from preprocessing.data_gen_utils import get_pitch_parselmouth,get_pitch_crepe
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import utils
import librosa
import torchcrepe
from infer import *
import logging
from infer_tools.infer_tool import *
import parselmouth
from parselmouth.praat import call
from IPython.display import Audio, display

_script = sys.argv[0]
_location = os.path.dirname(_script)
###FUNCTIONS
######################### RENDERING TAB FUNCTIONS
file_path = None
def choose_file(self, event):
        # Open the file selection dialog
        file_path = tkinter.filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        file_name = os.path.basename(file_path)
        self.select_wav_file.delete(0, tk.END)
        self.select_wav_file.insert(0, file_name)
        self.file_path = file_path


def render(self, file_path,svc_model):
    key = self.key_dropbox.get()
    key = int(key)
    pndm_speedup = self.speedup_drop.get()
    pndm_speedup = int(pndm_speedup)
    noise_step = self.noise_entry_value.get()
    add_noise_step = int(noise_step)
    use_crepe = bool(self.crepe.get())
    use_pe = bool(self.pe.get())
    threshold = float(self.tr_entry.get())
    print ("Crepe:",use_crepe)
    print ("Thre:",threshold)
    print("Noise:",add_noise_step)
    print("Speedup",pndm_speedup)
    print("key:",key)
    print("PE:",use_pe)
    wav_gen = tkinter.simpledialog.askstring("Input", "Enter the track name:", parent=Toplevel2)
    if not wav_gen.endswith('.wav'):
        wav_gen += '.wav'
    wav_fn = self.file_path
    demoaudio, sr = librosa.load(wav_fn)
    #Run
    f0_tst, f0_pred, audio = run_clip(svc_model, file_path=wav_fn, key=key, acc=pndm_speedup, use_crepe=use_crepe, use_pe=use_pe, thre=threshold,
                                        use_gt_mel=False, add_noise_step=add_noise_step, project_name=project_name, out_path=wav_gen)

    ##PRAAT
    formant_shift_ratio_str = self.Gflag.get()
    formant_shift_ratio = float(formant_shift_ratio_str)
    
    # If the formant shift ratio is not equal to 1.0, change the gender of the sound using parselmouth
    if formant_shift_ratio != 1.0:
        sound = parselmouth.Sound(wav_gen)
        print(wav_gen)
        Audio(data=sound.values, rate=sound.sampling_frequency)
        sound.get_power()
        sampling_rate = sound.sampling_frequency
        print(sampling_rate)
        resampled_sound = sound.resample(sampling_rate)
        print(resampled_sound)
        factor = formant_shift_ratio
        print(factor)
        manipulated_sound = call(sound, "Change gender", 75, 500, factor, 0, 1, 1)
        print(manipulated_sound)
        manipulated_sound.save("modified_audio.wav", "WAV")
        print("Gender correct!")
        # Play the modified sound using the default media player
    if formant_shift_ratio != 1.0:
        os.startfile("modified_audio.wav")
    else:
        os.startfile(wav_gen)
    
        #############
    
    
def load_model_function(self):
    filepath1 = tkinter.filedialog.askopenfilename(title = "Select CKPT File", filetypes=[("Checkpoint files", "*.ckpt")])
    if filepath1 == '':
        tkinter.messagebox.showerror("Error", "No CKPT file selected")
        return
    filepath2 = tkinter.filedialog.askopenfilename(title = "Select YAML File",filetypes=[("Yaml files", "*.yaml")])
    if filepath2 == '':
        tkinter.messagebox.showerror("Error", "No YAML file selected")
        return
    model_path = filepath1
    config_path = filepath2
    logging.getLogger('numba').setLevel(logging.WARNING)
    global project_name
    project_name = tkinter.simpledialog.askstring("Input", "Enter project name:") # Show a dialog box to input text
    if project_name == '':
        tkinter.messagebox.showerror("Error", "No Project Name")
        return
    # Use the input text and the value of hubert_gpu as arguments when creating an instance of the Svc class
    global svc_model
    hubert_gpu = bool(self.hubert_gpu.get())
    print (hubert_gpu)
    svc_model = Svc(project_name, config_path, hubert_gpu, model_path)
    # Extract the number of steps from the filename
    ckpt_filename = os.path.basename(model_path)
    ckpt_parts = ckpt_filename.split("_")
    steps = ckpt_parts[3]
    steps = steps.replace('.ckpt', '')
    steps = steps.replace("steps_", "")

    # Print the number of steps on the textbox
    self.consolebox.insert(tk.END, f"Model loaded, it has: {steps} steps")
    
    

####
def validate_speedup(new_text):
    if "-" in new_text:
        tkinter.messagebox.showerror("Error", "You can't set it lower than 0!")
        return False
    try:
        value = int(new_text)
        if value > 50:
            tkinter.messagebox.showwarning("Warning", "Any higher value than 50 may result in a significant quality loss.\nNote: if use_gt_mel below is enabled, make sure this value is lower than add_noise_step. This value should also be divisible by the number of diffusion steps.")
    except ValueError:
        pass
    return True


###noise
def check_noise_value(value):
    # If the value is an empty string, allow the change
    if value == "":
        return True
    # Try to convert the value to an integer
    try:
        value = int(value)
    except ValueError:
        # If the value cannot be converted to an integer,
        # assume it is invalid and return False
        return False
    # If the value is outside the allowed range,
    # display a warning message and return False
    if value < 0 or value > 1000:
        tkinter.messagebox.showerror("Error", "You can't set it higher than 1000 or lower than 0!")
        return False
    # If the value is within the allowed range, return True
    return True

def validatethreshold(value):
    if "-" in value:
        tkinter.messagebox.showerror("Error", "You can't set it lower than 0!")
        return False
    try:
        float_value = float(value)
        return True
    except ValueError:
        return False
def invalidthreshold(self):
    self.tr_entry_value.set("0.05")


    ##RENDER
    # noise_entry = self.noise_entry_value.get()
#threshold = self.tr_entry_value.get()
#speedup = self.speedup_value.get()
# file_path //(/ WAV)
# key = self.key_dropbox.get()
def waveget():
    wav_gen = tkinter.filedialog.asksaveasfilename(defaultextension='.wav', filetypes=[('Waveform Audio File Format', '*.wav')])



###WINDOW
class Toplevel2(tk.Tk):
    def __init__(self, top=None):
        super().__init__()
        self.initUI(top)
        self.resizable(False, False)

    def initUI(self, top):
        self.geometry("470x480")
        self.title("Diff-SVC Tool")

        self.top = top
        self.combobox = tk.StringVar()
        self.hubert_gpu = tk.IntVar()
        self.crepe = tk.IntVar()
        self.pe = tk.IntVar()
        self.gt_mel = tk.IntVar()

        self.TNotebook1 = ttk.Notebook(self.top)
        self.TNotebook1.place(relx=0.0, rely=0.0, relheight=1.012
                , relwidth=1.006)
        self.TNotebook1_t1 = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.TNotebook1_t1, padding=3)
        self.TNotebook1.tab(0, text='''Pre-Processing''', compound="left"
                ,underline='''-1''', )
        self.TNotebook1_t2 = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.TNotebook1_t2, padding=3)
        self.TNotebook1.tab(1, text='''Training''', compound="left"
                ,underline='''-1''', )
        self.TNotebook1_t3 = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.TNotebook1_t3, padding=3)
        self.TNotebook1.tab(2, text='''Rendering''', compound="left"
                ,underline='''-1''', )
        self.TNotebook1_t4 = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.TNotebook1_t4, padding=3)
        self.TNotebook1.tab(3, text='''Settings''', compound="left"
                ,underline='''-1''', )
############RENDERING TAB ########### RENDERING TAB ################ RENDERING TAB ##############
        self.load_model = ttk.Button(self.TNotebook1_t3, command=lambda: load_model_function(self))
        self.load_model.place(relx=0.084, rely=0.075, height=45, width=136)
        self.load_model.configure(text='''Load Model''')


        #self.choose_file_output = ttk.Button(self.TNotebook1_t3, command=waveget)
        #self.choose_file_output.place(relx=0.084, rely=0.3, height=45, width=136)
        #self.choose_file_output.configure(text='''Choose file output''')
        self.advanced_settings_label = ttk.Label(self.TNotebook1_t3)
        self.advanced_settings_label.place(relx=0.713, rely=0.0, height=18
                , width=105)
        self.advanced_settings_label.configure(text='''Advanced Settings''')
#######KEY DROPDOWN MENU ###### KEY DROPDOWN MENU ############ KEY DROPDOWN MENU #################
        global keysong
        self.key_dropbox = ttk.Combobox(self.TNotebook1_t3, values=list(range(-12, 13)))
        self.key_dropbox.set("0") ## Tomar valor con key = self.key_dropbox.get()
        self.combobox.set("0")
        self.key_dropbox.place(relx=0.713, rely=0.075, relheight=0.05
                , relwidth=0.218)
        self.key_dropbox.configure(state='readonly')
        self.key_dropbox.configure(textvariable=self.combobox)
        keysong = self.key_dropbox.get()


######## WAV SELECTOR ########### WAV SELECTOR ############# WAV SELECTOR ############# WAV SELECTOR
        self.select_wav_file = ttk.Entry(self.TNotebook1_t3)
        self.select_wav_file.place(relx=0.713, rely=0.2, relheight=0.053
                , relwidth=0.222)
        self.select_wav_file.insert(0, "Select a wav file")
        self.select_wav_file.bind("<Button-1>", lambda event: choose_file(self, event))


        ##RENDER
        self.start_rendering = ttk.Button(self.TNotebook1_t3, command=lambda: render(self, file_path, svc_model))
        self.start_rendering.place(relx=0.084, rely=0.525, height=45, width=136)
        self.start_rendering.configure(text='''Start Rendering''')

        ##CHECKBOX HUBERT
        self.checkbox = ttk.Checkbutton(self.TNotebook1_t3)
        self.checkbox.place(relx=0.524, rely=0.675, relwidth=0.189, relheight=0.0
                , height=21)
        self.checkbox.configure(variable=self.hubert_gpu)
        self.checkbox.configure(text='''Hubert GPU''')
        self.hubert_gpu.set(1)


        ##CREPE
        self.use_cp_check = ttk.Checkbutton(self.TNotebook1_t3)
        self.use_cp_check.place(relx=0.755, rely=0.75, relwidth=0.168
                , relheight=0.0, height=21)
        self.use_cp_check.configure(variable=self.crepe)
        self.use_cp_check.configure(text="Use 'crepe'")
        self.crepe.set(1)

        ##PE
        self.pe.set(1)
        self.use_pe_check = ttk.Checkbutton(self.TNotebook1_t3)
        self.use_pe_check.place(relx=0.755, rely=0.675, relwidth=0.147
                , relheight=0.0, height=21)
        self.use_pe_check.configure(variable=self.pe)
        self.use_pe_check.configure(text="Use 'pe'")
        

        ##GT MEL
        self.use_gp_check = ttk.Checkbutton(self.TNotebook1_t3)
        self.use_gp_check.place(relx=0.524, rely=0.75, relwidth=0.189
                , relheight=0.0, height=21)
        self.use_gp_check.configure(variable=self.gt_mel)
        self.use_gp_check.configure(text="Use 'gt_mel'")

        ##KEY
        self.keylabel = ttk.Label(self.TNotebook1_t3)
        self.keylabel.place(relx=0.587, rely=0.075, height=19, width=35)
        self.keylabel.configure(text='''Key:''')

        ##WAV
        self.wavlabel = ttk.Label(self.TNotebook1_t3)
        self.wavlabel.place(relx=0.587, rely=0.2, height=21, width=35)
        self.wavlabel.configure(text='''WAV:''')

        ##
        self.TSeparator2 = ttk.Separator(self.TNotebook1_t3)
        self.TSeparator2.place(relx=0.482, rely=0.025,  relheight=0.8)
        self.TSeparator2.configure(orient="vertical")
        self.consolebox = tk.Text(self.TNotebook1_t3)
        self.consolebox.place(relx=0.042, rely=0.825, relheight=0.11, relwidth=0.91)
        self.TSeparator3 = ttk.Separator(self.TNotebook1_t3)
        self.TSeparator3.place(relx=0.0, rely=0.825,  relwidth=1.006)

        ##speedup
        global speedup
        self.speedup_value = tk.StringVar()
        self.speedup_value.set(5)
        self.speedup_drop = tk.Entry(self.TNotebook1_t3, textvariable=self.speedup_value, validate="all", validatecommand=(self.register(validate_speedup), '%P'))
        self.speedup_drop.place(relx=0.713, rely=0.325, relheight=0.05
                , relwidth=0.218)
        pndm_speedup = self.speedup_drop.get()
        pndm_speedup = int(pndm_speedup)
        self.speeduplabel = ttk.Label(self.TNotebook1_t3)
        self.speeduplabel.place(relx=0.566, rely=0.325, height=19, width=55)
        self.speeduplabel.configure(text='''Speedup:''')
        ##CHECK UPDATES ON TERMINAL
        #def callback(name, index, mode):
        #        speedup = self.speedup_value.get()
        #        print(speedup)
        #self.speedup_value.trace("w", callback)



        ###noisesteps
        global noise_entry
        self.noise_entry_value = tk.StringVar()
        self.noise_entry_value.set(500)
        self.noisestepslabel = ttk.Label(self.TNotebook1_t3)
        self.noisestepslabel.place(relx=0.566, rely=0.45, height=19, width=65)
        self.noisestepslabel.configure(text='''Noise Step:''')
        self.noise_entry = tk.Entry(self.TNotebook1_t3, textvariable=self.noise_entry_value, validate="all", validatecommand=(self.register(check_noise_value), "%P"))
        noise_entry = self.noise_entry_value.get()
        self.noise_entry.place(relx=0.713, rely=0.45, relheight=0.053
                , relwidth=0.222)
        ##CHECK UPDATES ON TERMINAL
        #def callback(name, index, mode):
        #        noise_step = self.noise_entry_value.get()
        #        print(noise_step)
        #self.noise_entry_value.trace("w", callback)



        #Threshold
        global threshold
        self.tresholdlabel = ttk.Label(self.TNotebook1_t3)
        self.tresholdlabel.place(relx=0.566, rely=0.575, height=19, width=65)
        self.tresholdlabel.configure(text='''Threshold:''')
        self.tr_entry_value = tk.StringVar()
        self.tr_entry_value.set("0.05")
        self.tr_entry = ttk.Entry(self.TNotebook1_t3, textvariable=self.tr_entry_value, validate="all", validatecommand=(self.register(validatethreshold), "%P"), invalidcommand=invalidthreshold(self))
        self.tr_entry.place(relx=0.713, rely=0.575, relheight=0.053
                , relwidth=0.222)
        threshold = self.tr_entry_value.get()
        #def callback(name, index, mode):
        #        threshold = self.tr_entry_value.get()
        #        print(threshold)
        #self.tr_entry_value.trace("w", callback)

        ###GENDER FLAG
        self.genderflagtext = tk.Label(self.TNotebook1_t3)
        self.genderflagtext.place(relx=0.063, rely=0.283, height=31, width=74)
        self.genderflagtext.configure(anchor='w')
        self.genderflagtext.configure(text='''Gender Flag:''')
        self.Gflag = tk.Entry(self.TNotebook1_t3)
        self.Gflag.place(relx=0.231, rely=0.305, height=17, relwidth=0.155)

def start_up():
    print("hi")

if __name__ == '__main__':
    Toplevel2 = Toplevel2()
    Toplevel2.mainloop()



