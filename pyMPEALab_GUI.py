import sys, os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

# def resource_path(relative_path):
#     try:
#         base_path = sys._MEIPASS
#     except Exception:
#         base_path = os.path.abspath(".")
#     return os.path.join(base_path, relative_path)

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

import tkinter as tk
from tkinter import *
from tkinter import ttk

import numpy as np
import pandas as pd
from tensorflow.keras.layers import LeakyReLU, ReLU, Softmax
from pickle import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, RobustScaler

def start_gui():
    global window
    window = tk.Tk()
    window.title("pyMPEALab")
    window.iconbitmap(resource_path('pyMPEALab.ico'))

    # For Title
    l1 = Label(window, text="No. of Components", font='Helvetica 8 bold')
    l1.grid(row=0, column=0)

    global HEA
    global HEA
    global HEA_
    global elements
    elements = ['Ag', 'Al', 'Au', 'B', 'Be', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Fe', 'Gd', 'Hf', 'La', 'Li', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'Pd', 'Sc', 'Si', 'Sn', 'Sr', 'Ta', 'Ti', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']

    def tenth(hea):
        if int(num.get()) >9:
            global HEA10, e_10, elements10
            elements10 = [n for n in elements9 if n!= HEA9.get()]
            HEA10= ttk.Combobox(window, value=elements10, width=3, state='readonly')
            # HEA10.bind("<<ComboboxSelected>>", eleventh)
            HEA10.grid(row = 11, column=0)
            e_10 = StringVar()
            HEA_10 = Entry(window, textvariable=e_10, width=5)
            HEA_10.grid(row=11, column=1, pady = 5)

    def ninth(hea):
        if int(num.get()) >8:
            global HEA9, e_9, elements9
            elements9 = [n for n in elements8 if n!= HEA8.get()]
            HEA9= ttk.Combobox(window, value=elements9, width=3, state='readonly')
            HEA9.bind("<<ComboboxSelected>>", tenth)
            HEA9.grid(row = 10, column=0)
            e_9 = StringVar()
            HEA_9 = Entry(window, textvariable=e_9, width=5)
            HEA_9.grid(row=10, column=1, pady = 5)

    def eighth(hea):
        if int(num.get()) >7:
            global HEA8, e_8, elements8
            elements8 = [n for n in elements7 if n!= HEA7.get()]
            HEA8= ttk.Combobox(window, value=elements8, width=3, state='readonly')
            HEA8.bind("<<ComboboxSelected>>", ninth)
            HEA8.grid(row = 9, column=0)
            e_8 = StringVar()
            HEA_8 = Entry(window, textvariable=e_8, width=5)
            HEA_8.grid(row=9, column=1, pady = 5)

    def seventh(hea):
        if int(num.get()) >6:
            global HEA7, e_7, elements7
            elements7 = [n for n in elements6 if n!= HEA6.get()]
            HEA7= ttk.Combobox(window, value=elements7, width=3, state='readonly')
            HEA7.bind("<<ComboboxSelected>>", eighth)
            HEA7.grid(row = 8, column=0)
            e_7 = StringVar()
            HEA_7 = Entry(window, textvariable=e_7, width=5)
            HEA_7.grid(row=8, column=1, pady = 5)

    def sixth(hea):
        if int(num.get()) >5:
            global HEA6, e_6, elements6
            elements6 = [n for n in elements5 if n!= HEA5.get()]
            HEA6= ttk.Combobox(window, value=elements6, width=3, state='readonly')
            HEA6.bind("<<ComboboxSelected>>", seventh)
            HEA6.grid(row = 7, column=0)
            e_6 = StringVar()
            HEA_6 = Entry(window, textvariable=e_6, width=5)
            HEA_6.grid(row=7, column=1, pady = 5)

    def fifth(hea):
        if int(num.get()) > 4:
            global HEA5, e_5, elements5
            elements5 = [n for n in elements4 if n!= HEA4.get()]
            HEA5= ttk.Combobox(window, value=elements5, width=3, state='readonly')
            HEA5.bind("<<ComboboxSelected>>", sixth)
            HEA5.grid(row = 6, column=0)
            e_5 = StringVar()
            HEA_5 = Entry(window, textvariable=e_5, width=5)
            HEA_5.grid(row=6, column=1, pady = 5)

    def fourth(hea):
        if int(num.get()) > 3:
            global HEA4, e_4, elements4
            elements4 = [n for n in elements3 if n!= HEA3.get()]
            HEA4= ttk.Combobox(window, value=elements4, width=3, state='readonly')
            HEA4.bind("<<ComboboxSelected>>", fifth)
            HEA4.grid(row = 5, column=0)
            e_4 = StringVar()
            HEA_4 = Entry(window, textvariable=e_4, width=5)
            HEA_4.grid(row=5, column=1, pady = 5)

    def third(hea):
        if int(num.get()) > 2:
            global HEA3, e_3, elements3
            elements3 = [n for n in elements2 if n!= HEA2.get()]
            HEA3= ttk.Combobox(window, value=elements3, width=3, state='readonly')
            HEA3.bind("<<ComboboxSelected>>", fourth)
            HEA3.grid(row = 4, column=0)
            e_3 = StringVar()
            HEA_3 = Entry(window, textvariable=e_3, width=5)
            HEA_3.grid(row=4, column=1, pady = 5)
        
    def second(hea):
        if int(num.get()) < 11:
            global HEA2, e_2, elements2
            elements2 = [n for n in elements if n != HEA1.get()]
            HEA2= ttk.Combobox(window, value=elements2, width=3, state='readonly')
            HEA2.bind("<<ComboboxSelected>>", third)
            HEA2.grid(row = 3, column=0)
            e_2 = StringVar()
            HEA_2 = Entry(window, textvariable=e_2, width=5)
            HEA_2.grid(row=3, column=1, pady = 5)

    def first(no):
        if int(num.get()) < 11:
            global HEA1, e_1
            e1 = StringVar()
            HEA1= ttk.Combobox(window, value=elements, width=3, state='readonly')
            HEA1.bind("<<ComboboxSelected>>", second)
            HEA1.grid(row = 2, column=0)
            e_1 = StringVar()
            HEA_1 = Entry(window, textvariable=e_1, width=5)
            HEA_1.grid(row=2, column=1, pady = 5)

    global num
    component_number = [2,3,4,5,6,7,8,9,10]
    num = ttk.Combobox(window, value=component_number, width=2, state= 'readonly')
    num.bind("<<ComboboxSelected>>", first)
    num.grid(row = 0, column=1)

    HEAs = Label(window, text='Elements', anchor='w', width = 10, font='Helvetica 10 bold')
    HEAs.grid(row=1, column=0)
    Compositions = Label(window, text='Composition', anchor='w', width = 10, font='Helvetica 10 bold')
    Compositions.grid(row=1, column=1)
    Properties = Label(window, text='Properties', anchor='center', width = 10, font='Helvetica 10 bold', fg='red')
    Properties.grid(row=1, column=2)
    Valuess = Label(window, text='Values', anchor='center', width = 10, font='Helvetica 10 bold', fg='red')
    Valuess.grid(row=1, column=3)

    def properties():
        HEA = []
        HEA_ = []

        if int(num.get()) > 0:
            HEA.append(HEA1.get()), HEA_.append(e_1.get())
            if int(num.get()) > 1:
                HEA.append(HEA2.get()), HEA_.append(e_2.get())
                if int(num.get()) > 2:
                    HEA.append(HEA3.get()),  HEA_.append(e_3.get())
                    if int(num.get()) > 3:
                        HEA.append(HEA4.get()),  HEA_.append(e_4.get())
                        if int(num.get()) > 4:
                            HEA.append(HEA5.get()),  HEA_.append(e_5.get())
                            if int(num.get()) > 5:
                                HEA.append(HEA6.get()),  HEA_.append(e_6.get())
                                if int(num.get()) > 6:
                                    HEA.append(HEA7.get()),  HEA_.append(e_7.get())
                                    if int(num.get()) > 7:
                                        HEA.append(HEA8.get()),  HEA_.append(e_8.get())
                                        if int(num.get()) > 8:
                                            HEA.append(HEA9.get()),  HEA_.append(e_9.get())
                                            if int(num.get()) > 9:
                                                HEA.append(HEA10.get()),  HEA_.append(e_10.get())

        global full_hea
        full = []
        for a in range(int(num.get())):
            if HEA_[a] != '1':
                full.append(HEA[a]), full.append(HEA_[a])
            else:
                full.append(HEA[a])
        full_hea = ''.join(full)

        HEA_ = [float(i) for i in HEA_]
        total_mole = sum(HEA_)
        mole_fraction = np.divide(HEA_, total_mole)
        elements = ['Ag', 'Al', 'Au', 'B', 'Be', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Fe', 'Gd', 'Hf', 'La', 'Li', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'Pd', 'Sc', 'Si', 'Sn', 'Sr', 'Ta', 'Ti', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']
        composition = {'Ag':0, 'Al':0, 'Au':0, 'B':0, 'Be':0, 'Ca':0, 'Ce':0, 'Co':0, 'Cr':0, 'Cu':0, 'Fe':0, 'Gd':0, 'Hf':0, 'La':0, 'Li':0, 'Mg':0, 'Mn':0, 'Mo':0, 'Nb':0, 'Nd':0, 'Ni':0, 'Pd':0, 'Sc':0, 'Si':0, 'Sn':0, 'Sr':0, 'Ta':0, 'Ti':0, 'V':0, 'W':0, 'Y':0, 'Yb':0, 'Zn':0, 'Zr':0}

        for i in range(len(HEA)):
            if HEA[i] in composition:
                composition.update({HEA[i]:mole_fraction[i]})
        composition_list = list(composition.values())
        attributes = list(composition.values())
        radius_data = {'Ag':1.447, 'Al':1.4317, 'Au':1.442, 'B':0.82, 'Be':1.128, 'Ca':1.976, 'Ce':1.824, 'Co':1.251, 'Cr':1.2491, 'Cu':1.278, 'Fe':1.2412, 'Gd':1.8013, 'Hf':1.5775, 'La':1.879, 'Li':1.5194, 'Mg':1.6013, 'Mn':1.35, 'Mo':1.3626, 'Nb':1.429, 'Nd':1.64, 'Ni': 1.2459, 'Pd':1.3754, 'Sc':1.641, 'Si':1.153, 'Sn':1.62, 'Sr':2.152, 'Ta':1.43, 'Ti':1.4615, 'V':1.316, 'W':1.367, 'Y':1.8015, 'Yb':1.7, 'Zn':1.3945, 'Zr':1.6025}

        r_i =[]
        for a in HEA:
            if a in radius_data:
                r_i.append(radius_data.get(a))

        r_bar = sum(np.multiply(mole_fraction, r_i))
        term = (1-np.divide(r_i, r_bar))**2

        atomic_size_difference = sum(np.multiply(mole_fraction, term))**0.5

        attributes.append(atomic_size_difference)

        H_ab_1 = {'AgAl': -4.0, 'AgAu': -5.535, 'AgB': 3.938, 'AgBe': 5.809, 'AgCa': -28.49, 'AgCe': -30.0, 'AgCo': 18.855999999999998, 'AgCr': 26.728, 'AgCu': 2.0, 'AgFe': 28.0, 'AgGd': -28.732, 'AgHf': -12.41, 'AgLa': -30.0, 'AgLi': -15.828, 'AgMg': -10.0, 'AgMn': 12.475, 'AgMo': 36.982, 'AgNb': 16.160999999999998, 'AgNd': -28.768, 'AgNi': 15.267999999999999, 'AgPd': -4.0, 'AgSc': -27.906999999999996, 'AgSi': -20.0, 'AgSn': -4.6739999999999995, 'AgSr': -27.111, 'AgTa': 14.83, 'AgTi': -1.571, 'AgV': 16.651, 'AgW': 42.714, 'AgY': -29.0, 'AgYb': -26.989, 'AgZn': -4.828, 'AgZr': -20.023, 'AlAu': -22.0, 'AlB': 0.0, 'AlBe': 0.044000000000000004, 'AlCa': -20.0, 'AlCe': -38.0, 'AlCo': -19.0, 'AlCr': -10.0, 'AlCu': -1.0, 'AlFe': -11.0, 'AlGd': -38.0, 'AlHf': -39.0, 'AlLa': -38.0, 'AlLi': -3.384, 'AlMg': -2.0, 'AlMn': -19.0, 'AlMo': -5.0, 'AlNb': -18.0, 'AlNd': -38.0, 'AlNi': -22.0, 'AlPd': -46.0, 'AlSc': -45.215, 'AlSi': -19.0, 'AlSn': 4.124, 'AlSr': -6.382000000000001, 'AlTa': -30.750999999999998, 'AlTi': -30.0, 'AlV': -16.0, 'AlW': -13.796, 'AlY': -38.0, 'AlYb': -22.779, 'AlZn': 1.0, 'AlZr': -44.0, 'AuB': -2.0, 'AuBe': -0.634, 'AuCa': -60.338, 'AuCe': -70.79899999999999, 'AuCo': 7.138999999999999, 'AuCr': -0.209, 'AuCu': -9.0, 'AuFe': 8.0, 'AuGd': -74.0, 'AuHf': -61.894, 'AuLa': -73.0, 'AuLi': -38.113, 'AuMg': -32.0, 'AuMn': -11.163, 'AuMo': 3.432, 'AuNb': -31.55, 'AuNd': -70.957, 'AuNi': 7.099, 'AuPd': 0.0, 'AuSc': -72.671, 'AuSi': -30.0, 'AuSn': -13.872, 'AuSr': -58.203, 'AuTa': -31.795, 'AuTi': -46.62, 'AuV': -18.977, 'AuW': 11.415, 'AuY': -71.308, 'AuYb': -57.325, 'AuZn': -16.0, 'AuZr': -72.061, 'BBe': 0.0, 'BCa': -12.227, 'BCe': -52.091, 'BCo': -24.0, 'BCr': -31.0, 'BCu': 0.0, 'BFe': -26.0, 'BGd': -50.0, 'BHf': -66.0, 'BLa': -47.0, 'BLi': -5.232, 'BMg': -3.59, 'BMn': -32.0, 'BMo': -34.0, 'BNb': -54.0, 'BNd': -49.0, 'BNi': -24.0, 'BPd': -24.0, 'BSc': -55.0, 'BSi': -14.0, 'BSn': 18.0, 'BSr': -8.017999999999999, 'BTa': -54.0, 'BTi': -58.0, 'BV': -42.0, 'BW': -31.0, 'BY': -50.0, 'BYb': -22.805999999999997, 'BZn': 4.073, 'BZr': -71.0, 'BeCa': -10.789000000000001, 'BeCe': -29.232, 'BeCo': -5.76, 'BeCr': -8.997, 'BeCu': -1.642, 'BeFe': -4.0, 'BeGd': -30.647, 'BeHf': -37.0, 'BeLa': -28.002, 'BeLi': -4.409, 'BeMg': -2.86, 'BeMn': -12.014000000000001, 'BeMo': -8.652999999999999, 'BeNb': -25.0, 'BeNd': -29.927, 'BeNi': -6.068, 'BePd': -9.967, 'BeSc': -36.0, 'BeSi': -15.0, 'BeSn': 14.144, 'BeSr': -6.9910000000000005, 'BeTa': -25.594, 'BeTi': -30.0, 'BeV': -17.881, 'BeW': -5.2620000000000005, 'BeY': -30.647, 'BeYb': -13.238, 'BeZn': 3.412, 'BeZr': -43.0, 'CaCe': 1.9380000000000002, 'CaCo': 2.0, 'CaCr': 29.689, 'CaCu': -13.0, 'CaFe': 25.0, 'CaGd': 3.842, 'CaHf': 29.846, 'CaLa': 8.0, 'CaLi': -0.723, 'CaMg': -6.0, 'CaMn': 11.565999999999999, 'CaMo': 47.707, 'CaNb': 53.976000000000006, 'CaNd': 2.885, 'CaNi': -7.0, 'CaPd': -70.44, 'CaSc': 10.847999999999999, 'CaSi': -22.627, 'CaSn': -31.739, 'CaSr': 0.517, 'CaTa': 50.792, 'CaTi': 33.897, 'CaV': 35.225, 'CaW': 48.458, 'CaY': 3.842, 'CaYb': -3.352, 'CaZn': -22.0, 'CaZr': 27.444000000000003, 'CeCo': -18.0, 'CeCr': 15.0, 'CeCu': -21.0, 'CeFe': 3.0, 'CeGd': 0.09, 'CeHf': 13.713, 'CeLa': 0.04, 'CeLi': 6.683, 'CeMg': -7.0, 'CeMn': 1.0, 'CeMo': 28.253, 'CeNb': 34.0, 'CeNd': 0.023, 'CeNi': -28.0, 'CePd': -81.169, 'CeSc': 1.599, 'CeSi': -64.329, 'CeSn': -63.01, 'CeSr': 7.563, 'CeTa': 30.641, 'CeTi': 17.652, 'CeV': 20.0, 'CeW': 28.165, 'CeY': 0.09, 'CeYb': 8.0, 'CeZn': -31.0, 'CeZr': 11.344000000000001, 'CoCr': -4.0, 'CoCu': 6.372999999999999, 'CoFe': -1.0, 'CoGd': -22.0, 'CoHf': -35.0, 'CoLa': -17.0, 'CoLi': 8.687000000000001, 'CoMg': 0.853, 'CoMn': -5.0, 'CoMo': -5.0, 'CoNb': -25.0, 'CoNd': -19.586, 'CoNi': 0.0, 'CoPd': -1.0, 'CoSc': -29.328000000000003, 'CoSi': -38.0, 'CoSn': -11.134, 'CoSr': 2.178, 'CoTa': -24.0, 'CoTi': -28.0, 'CoV': -14.0, 'CoW': -1.0, 'CoY': -22.0, 'CoYb': 2.21, 'CoZn': -12.061, 'CoZr': -41.0, 'CrCu': 12.385, 'CrFe': -1.0, 'CrGd': 11.073, 'CrHf': -9.148, 'CrLa': 17.0, 'CrLi': 35.441, 'CrMg': 21.854, 'CrMn': 2.141, 'CrMo': 0.0, 'CrNb': -7.122000000000001, 'CrNd': 12.642000000000001, 'CrNi': -7.0, 'CrPd': -15.0, 'CrSc': 0.6779999999999999, 'CrSi': -37.0, 'CrSn': -1.632, 'CrSr': 39.624, 'CrTa': -6.653, 'CrTi': -7.348, 'CrV': -1.9369999999999998, 'CrW': 0.951, 'CrY': 11.073, 'CrYb': 36.223, 'CrZn': -2.247, 'CrZr': -12.0, 'CuFe': 13.0, 'CuGd': -22.0, 'CuHf': -17.0, 'CuLa': -21.0, 'CuLi': -4.737, 'CuMg': -3.0, 'CuMn': 4.0, 'CuMo': 18.45, 'CuNb': 3.0, 'CuNd': -22.0, 'CuNi': 4.0, 'CuPd': -14.0, 'CuSc': -23.522, 'CuSi': -19.0, 'CuSn': -4.177, 'CuSr': -9.0, 'CuTa': 1.808, 'CuTi': -9.0, 'CuV': 5.0, 'CuW': 22.326999999999998, 'CuY': -22.0, 'CuYb': -12.0, 'CuZn': 1.0, 'CuZr': -23.0, 'FeGd': -1.0, 'FeHf': -21.0, 'FeLa': 5.0, 'FeLi': 26.785, 'FeMg': 15.777999999999999, 'FeMn': 0.0, 'FeMo': -2.0, 'FeNb': -16.0, 'FeNd': 1.0, 'FeNi': -2.0, 'FePd': -4.0, 'FeSc': -11.0, 'FeSi': -35.0, 'FeSn': 11.0, 'FeSr': 26.099, 'FeTa': -15.0, 'FeTi': -17.0, 'FeV': -7.0, 'FeW': 0.0, 'FeY': -0.895, 'FeYb': 24.44, 'FeZn': -3.389, 'FeZr': -25.0, 'GdHf': 11.41, 'GdLa': 0.25, 'GdLi': 7.907, 'GdMg': -7.7989999999999995, 'GdMn': -1.4709999999999999, 'GdMo': 24.0, 'GdNb': 29.302, 'GdNd': 0.022000000000000002, 'GdNi': -31.0, 'GdPd': -82.235, 'GdSc': 0.934, 'GdSi': -73.0, 'GdSn': -61.261, 'GdSr': 9.86, 'GdTa': 26.945, 'GdTi': 15.03, 'GdV': 16.512, 'GdW': 23.631999999999998, 'GdY': 0.0, 'GdYb': 10.22, 'GdZn': -35.84, 'GdZr': 9.0, 'HfLa': 15.335, 'HfLi': 30.675, 'HfMg': 7.279, 'HfMn': -11.790999999999999, 'HfMo': -3.8710000000000004, 'HfNb': 4.0, 'HfNd': 12.495999999999999, 'HfNi': -42.0, 'HfPd': -79.275, 'HfSc': 5.087, 'HfSi': -77.0, 'HfSn': -49.183, 'HfSr': 40.378, 'HfTa': 3.0, 'HfTi': 0.14800000000000002, 'HfV': -2.0, 'HfW': -6.347, 'HfY': 11.41, 'HfYb': 37.754, 'HfZn': -31.405, 'HfZr': 0.0, 'LaLi': 5.88, 'LaMg': -7.0, 'LaMn': 3.0, 'LaMo': 31.0, 'LaNb': 36.0, 'LaNd': 0.124, 'LaNi': -27.0, 'LaPd': -79.792, 'LaSc': 2.131, 'LaSi': -63.70399999999999, 'LaSn': -63.687, 'LaSr': 14.0, 'LaTa': 33.183, 'LaTi': 19.489, 'LaV': 21.756999999999998, 'LaW': 31.379, 'LaY': 0.25, 'LaYb': 7.401, 'LaZn': -31.0, 'LaZr': 13.0, 'LiMg': -0.33, 'LiMn': 19.769000000000002, 'LiMo': 50.726000000000006, 'LiNb': 52.221000000000004, 'LiNd': 7.276, 'LiNi': 0.963, 'LiPd': -41.93899999999999, 'LiSc': 12.562000000000001, 'LiSi': -13.034, 'LiSn': -18.26, 'LiSr': -0.27699999999999997, 'LiTa': 49.708999999999996, 'LiTi': 34.854, 'LiV': 38.228, 'LiW': 52.198, 'LiY': 7.907, 'LiYb': -0.6759999999999999, 'LiZn': -6.893, 'LiZr': 27.840999999999998, 'MgMn': 8.038, 'MgMo': 34.080999999999996, 'MgNb': 29.51, 'MgNd': -6.0, 'MgNi': -4.0, 'MgPd': -40.0, 'MgSc': -5.023, 'MgSi': -9.029, 'MgSn': -9.0, 'MgSr': -1.419, 'MgTa': 24.496, 'MgTi': 16.0, 'MgV': 21.11, 'MgW': 36.334, 'MgY': -6.0, 'MgYb': -6.292999999999999, 'MgZn': -4.0, 'MgZr': 3.228, 'MnMo': 4.945, 'MnNb': -4.0, 'MnNd': -0.348, 'MnNi': -8.0, 'MnPd': -23.0, 'MnSc': -8.363999999999999, 'MnSi': -45.0, 'MnSn': -18.439, 'MnSr': 18.601, 'MnTa': -3.8, 'MnTi': -8.094, 'MnV': -0.708, 'MnW': 6.2989999999999995, 'MnY': -1.4340000000000002, 'MnYb': 18.299, 'MnZn': -13.345999999999998, 'MnZr': -15.0, 'MoNb': -6.0, 'MoNd': 26.029, 'MoNi': -7.0, 'MoPd': -14.457, 'MoSc': 10.507, 'MoSi': -35.0, 'MoSn': 7.337999999999999, 'MoSr': 59.291000000000004, 'MoTa': -4.8580000000000005, 'MoTi': -3.5069999999999997, 'MoV': 0.01, 'MoW': -0.221, 'MoY': 24.004, 'MoYb': 54.835, 'MoZn': 4.2330000000000005, 'MoZr': -6.0, 'NbNd': 31.11, 'NbNi': -30.0, 'NbPd': -53.0, 'NbSc': 17.614, 'NbSi': -56.0, 'NbSn': -1.0, 'NbSr': 66.48, 'NbTa': 0.0, 'NbTi': 2.0, 'NbV': -1.0, 'NbW': -8.0, 'NbY': 29.302, 'NbYb': 61.126000000000005, 'NbZn': -9.193999999999999, 'NbZr': 4.0, 'NdNi': -30.0, 'NdPd': -81.59899999999999, 'NdSc': 1.238, 'NdSi': -64.374, 'NdSn': -62.01, 'NdSr': 8.696, 'NdTa': 28.69, 'NdTi': 16.271, 'NdV': 17.975, 'NdW': 25.796, 'NdY': 0.022000000000000002, 'NdYb': 9.363999999999999, 'NdZn': -35.958, 'NdZr': 10.269, 'NiPd': 0.0, 'NiSc': -37.938, 'NiSi': -40.0, 'NiSn': -15.136, 'NiSr': -7.983, 'NiTa': -29.0, 'NiTi': -35.0, 'NiV': -18.0, 'NiW': -3.0, 'NiY': -31.0, 'NiYb': -7.341, 'NiZn': -15.802, 'NiZr': -49.0, 'PdSc': -85.48899999999999, 'PdSi': -55.0, 'PdSn': -46.083999999999996, 'PdSr': -67.265, 'PdTa': -52.0, 'PdTi': -65.0, 'PdV': -35.111, 'PdW': -6.433, 'PdY': -82.235, 'PdYb': -61.787, 'PdZn': -40.86, 'PdZr': -91.0, 'ScSi': -64.891, 'ScSn': -54.997, 'ScSr': 17.991, 'ScTa': 15.706, 'ScTi': 7.4079999999999995, 'ScV': 7.037000000000001, 'ScW': 9.137, 'ScY': 0.934, 'ScYb': 16.421, 'ScZn': -34.21, 'ScZr': 4.0, 'SiSn': -11.0, 'SiSr': -20.004, 'SiTa': -56.0, 'SiTi': -66.0, 'SiV': -48.0, 'SiW': -31.0, 'SiY': -64.58800000000001, 'SiYb': -36.903, 'SiZn': -0.845, 'SiZr': -84.0, 'SnSr': -32.944, 'SnTa': -16.180999999999997, 'SnTi': -34.341, 'SnV': -13.04, 'SnW': 13.159, 'SnY': -61.261, 'SnYb': -48.28, 'SnZn': 1.0, 'SnZr': -43.0, 'SrTa': 62.966, 'SrTi': 44.06100000000001, 'SrV': 44.805, 'SrW': 60.364, 'SrY': 9.86, 'SrYb': -2.8310000000000004, 'SrZn': -11.969000000000001, 'SrZr': 37.849000000000004, 'TaTi': 1.0, 'TaV': -1.0070000000000001, 'TaW': -7.289, 'TaY': 26.945, 'TaYb': 57.99100000000001, 'TaZn': -10.401, 'TaZr': 3.0, 'TiV': -2.0, 'TiW': -5.636, 'TiY': 15.03, 'TiYb': 41.121, 'TiZn': -22.324, 'TiZr': 0.0, 'VW': -0.7929999999999999, 'VY': 16.512, 'VYb': 41.961999999999996, 'VZn': -8.834, 'VZr': -4.0, 'WY': 23.631999999999998, 'WYb': 55.691, 'WZn': 7.684, 'WZr': -9.0, 'YYb': 10.22, 'YZn': -35.84, 'YZr': 9.0, 'YbZn': -22.791, 'YbZr': 35.4, 'ZnZr': -36.735}
        H_ab_2 = {'AlAg': -4.0, 'AuAg': -5.535, 'BAg': 3.938, 'BeAg': 5.809, 'CaAg': -28.49, 'CeAg': -30.0, 'CoAg': 18.855999999999998, 'CrAg': 26.728, 'CuAg': 2.0, 'FeAg': 28.0, 'GdAg': -28.732, 'HfAg': -12.41, 'LaAg': -30.0, 'LiAg': -15.828, 'MgAg': -10.0, 'MnAg': 12.475, 'MoAg': 36.982, 'NbAg': 16.160999999999998, 'NdAg': -28.768, 'NiAg': 15.267999999999999, 'PdAg': -4.0, 'ScAg': -27.906999999999996, 'SiAg': -20.0, 'SnAg': -4.6739999999999995, 'SrAg': -27.111, 'TaAg': 14.83, 'TiAg': -1.571, 'VAg': 16.651, 'WAg': 42.714, 'YAg': -29.0, 'YbAg': -26.989, 'ZnAg': -4.828, 'ZrAg': -20.023, 'AuAl': -22.0, 'BAl': 0.0, 'BeAl': 0.044000000000000004, 'CaAl': -20.0, 'CeAl': -38.0, 'CoAl': -19.0, 'CrAl': -10.0, 'CuAl': -1.0, 'FeAl': -11.0, 'GdAl': -38.0, 'HfAl': -39.0, 'LaAl': -38.0, 'LiAl': -3.384, 'MgAl': -2.0, 'MnAl': -19.0, 'MoAl': -5.0, 'NbAl': -18.0, 'NdAl': -38.0, 'NiAl': -22.0, 'PdAl': -46.0, 'ScAl': -45.215, 'SiAl': -19.0, 'SnAl': 4.124, 'SrAl': -6.382000000000001, 'TaAl': -30.750999999999998, 'TiAl': -30.0, 'VAl': -16.0, 'WAl': -13.796, 'YAl': -38.0, 'YbAl': -22.779, 'ZnAl': 1.0, 'ZrAl': -44.0, 'BAu': -2.0, 'BeAu': -0.634, 'CaAu': -60.338, 'CeAu': -70.79899999999999, 'CoAu': 7.138999999999999, 'CrAu': -0.209, 'CuAu': -9.0, 'FeAu': 8.0, 'GdAu': -74.0, 'HfAu': -61.894, 'LaAu': -73.0, 'LiAu': -38.113, 'MgAu': -32.0, 'MnAu': -11.163, 'MoAu': 3.432, 'NbAu': -31.55, 'NdAu': -70.957, 'NiAu': 7.099, 'PdAu': 0.0, 'ScAu': -72.671, 'SiAu': -30.0, 'SnAu': -13.872, 'SrAu': -58.203, 'TaAu': -31.795, 'TiAu': -46.62, 'VAu': -18.977, 'WAu': 11.415, 'YAu': -71.308, 'YbAu': -57.325, 'ZnAu': -16.0, 'ZrAu': -72.061, 'BeB': 0.0, 'CaB': -12.227, 'CeB': -52.091, 'CoB': -24.0, 'CrB': -31.0, 'CuB': 0.0, 'FeB': -26.0, 'GdB': -50.0, 'HfB': -66.0, 'LaB': -47.0, 'LiB': -5.232, 'MgB': -3.59, 'MnB': -32.0, 'MoB': -34.0, 'NbB': -54.0, 'NdB': -49.0, 'NiB': -24.0, 'PdB': -24.0, 'ScB': -55.0, 'SiB': -14.0, 'SnB': 18.0, 'SrB': -8.017999999999999, 'TaB': -54.0, 'TiB': -58.0, 'VB': -42.0, 'WB': -31.0, 'YB': -50.0, 'YbB': -22.805999999999997, 'ZnB': 4.073, 'ZrB': -71.0, 'CaBe': -10.789000000000001, 'CeBe': -29.232, 'CoBe': -5.76, 'CrBe': -8.997, 'CuBe': -1.642, 'FeBe': -4.0, 'GdBe': -30.647, 'HfBe': -37.0, 'LaBe': -28.002, 'LiBe': -4.409, 'MgBe': -2.86, 'MnBe': -12.014000000000001, 'MoBe': -8.652999999999999, 'NbBe': -25.0, 'NdBe': -29.927, 'NiBe': -6.068, 'PdBe': -9.967, 'ScBe': -36.0, 'SiBe': -15.0, 'SnBe': 14.144, 'SrBe': -6.9910000000000005, 'TaBe': -25.594, 'TiBe': -30.0, 'VBe': -17.881, 'WBe': -5.2620000000000005, 'YBe': -30.647, 'YbBe': -13.238, 'ZnBe': 3.412, 'ZrBe': -43.0, 'CeCa': 1.9380000000000002, 'CoCa': 2.0, 'CrCa': 29.689, 'CuCa': -13.0, 'FeCa': 25.0, 'GdCa': 3.842, 'HfCa': 29.846, 'LaCa': 8.0, 'LiCa': -0.723, 'MgCa': -6.0, 'MnCa': 11.565999999999999, 'MoCa': 47.707, 'NbCa': 53.976000000000006, 'NdCa': 2.885, 'NiCa': -7.0, 'PdCa': -70.44, 'ScCa': 10.847999999999999, 'SiCa': -22.627, 'SnCa': -31.739, 'SrCa': 0.517, 'TaCa': 50.792, 'TiCa': 33.897, 'VCa': 35.225, 'WCa': 48.458, 'YCa': 3.842, 'YbCa': -3.352, 'ZnCa': -22.0, 'ZrCa': 27.444000000000003, 'CoCe': -18.0, 'CrCe': 15.0, 'CuCe': -21.0, 'FeCe': 3.0, 'GdCe': 0.09, 'HfCe': 13.713, 'LaCe': 0.04, 'LiCe': 6.683, 'MgCe': -7.0, 'MnCe': 1.0, 'MoCe': 28.253, 'NbCe': 34.0, 'NdCe': 0.023, 'NiCe': -28.0, 'PdCe': -81.169, 'ScCe': 1.599, 'SiCe': -64.329, 'SnCe': -63.01, 'SrCe': 7.563, 'TaCe': 30.641, 'TiCe': 17.652, 'VCe': 20.0, 'WCe': 28.165, 'YCe': 0.09, 'YbCe': 8.0, 'ZnCe': -31.0, 'ZrCe': 11.344000000000001, 'CrCo': -4.0, 'CuCo': 6.372999999999999, 'FeCo': -1.0, 'GdCo': -22.0, 'HfCo': -35.0, 'LaCo': -17.0, 'LiCo': 8.687000000000001, 'MgCo': 0.853, 'MnCo': -5.0, 'MoCo': -5.0, 'NbCo': -25.0, 'NdCo': -19.586, 'NiCo': 0.0, 'PdCo': -1.0, 'ScCo': -29.328000000000003, 'SiCo': -38.0, 'SnCo': -11.134, 'SrCo': 2.178, 'TaCo': -24.0, 'TiCo': -28.0, 'VCo': -14.0, 'WCo': -1.0, 'YCo': -22.0, 'YbCo': 2.21, 'ZnCo': -12.061, 'ZrCo': -41.0, 'CuCr': 12.385, 'FeCr': -1.0, 'GdCr': 11.073, 'HfCr': -9.148, 'LaCr': 17.0, 'LiCr': 35.441, 'MgCr': 21.854, 'MnCr': 2.141, 'MoCr': 0.0, 'NbCr': -7.122000000000001, 'NdCr': 12.642000000000001, 'NiCr': -7.0, 'PdCr': -15.0, 'ScCr': 0.6779999999999999, 'SiCr': -37.0, 'SnCr': -1.632, 'SrCr': 39.624, 'TaCr': -6.653, 'TiCr': -7.348, 'VCr': -1.9369999999999998, 'WCr': 0.951, 'YCr': 11.073, 'YbCr': 36.223, 'ZnCr': -2.247, 'ZrCr': -12.0, 'FeCu': 13.0, 'GdCu': -22.0, 'HfCu': -17.0, 'LaCu': -21.0, 'LiCu': -4.737, 'MgCu': -3.0, 'MnCu': 4.0, 'MoCu': 18.45, 'NbCu': 3.0, 'NdCu': -22.0, 'NiCu': 4.0, 'PdCu': -14.0, 'ScCu': -23.522, 'SiCu': -19.0, 'SnCu': -4.177, 'SrCu': -9.0, 'TaCu': 1.808, 'TiCu': -9.0, 'VCu': 5.0, 'WCu': 22.326999999999998, 'YCu': -22.0, 'YbCu': -12.0, 'ZnCu': 1.0, 'ZrCu': -23.0, 'GdFe': -1.0, 'HfFe': -21.0, 'LaFe': 5.0, 'LiFe': 26.785, 'MgFe': 15.777999999999999, 'MnFe': 0.0, 'MoFe': -2.0, 'NbFe': -16.0, 'NdFe': 1.0, 'NiFe': -2.0, 'PdFe': -4.0, 'ScFe': -11.0, 'SiFe': -35.0, 'SnFe': 11.0, 'SrFe': 26.099, 'TaFe': -15.0, 'TiFe': -17.0, 'VFe': -7.0, 'WFe': 0.0, 'YFe': -0.895, 'YbFe': 24.44, 'ZnFe': -3.389, 'ZrFe': -25.0, 'HfGd': 11.41, 'LaGd': 0.25, 'LiGd': 7.907, 'MgGd': -7.7989999999999995, 'MnGd': -1.4709999999999999, 'MoGd': 24.0, 'NbGd': 29.302, 'NdGd': 0.022000000000000002, 'NiGd': -31.0, 'PdGd': -82.235, 'ScGd': 0.934, 'SiGd': -73.0, 'SnGd': -61.261, 'SrGd': 9.86, 'TaGd': 26.945, 'TiGd': 15.03, 'VGd': 16.512, 'WGd': 23.631999999999998, 'YGd': 0.0, 'YbGd': 10.22, 'ZnGd': -35.84, 'ZrGd': 9.0, 'LaHf': 15.335, 'LiHf': 30.675, 'MgHf': 7.279, 'MnHf': -11.790999999999999, 'MoHf': -3.8710000000000004, 'NbHf': 4.0, 'NdHf': 12.495999999999999, 'NiHf': -42.0, 'PdHf': -79.275, 'ScHf': 5.087, 'SiHf': -77.0, 'SnHf': -49.183, 'SrHf': 40.378, 'TaHf': 3.0, 'TiHf': 0.14800000000000002, 'VHf': -2.0, 'WHf': -6.347, 'YHf': 11.41, 'YbHf': 37.754, 'ZnHf': -31.405, 'ZrHf': 0.0, 'LiLa': 5.88, 'MgLa': -7.0, 'MnLa': 3.0, 'MoLa': 31.0, 'NbLa': 36.0, 'NdLa': 0.124, 'NiLa': -27.0, 'PdLa': -79.792, 'ScLa': 2.131, 'SiLa': -63.70399999999999, 'SnLa': -63.687, 'SrLa': 14.0, 'TaLa': 33.183, 'TiLa': 19.489, 'VLa': 21.756999999999998, 'WLa': 31.379, 'YLa': 0.25, 'YbLa': 7.401, 'ZnLa': -31.0, 'ZrLa': 13.0, 'MgLi': -0.33, 'MnLi': 19.769000000000002, 'MoLi': 50.726000000000006, 'NbLi': 52.221000000000004, 'NdLi': 7.276, 'NiLi': 0.963, 'PdLi': -41.93899999999999, 'ScLi': 12.562000000000001, 'SiLi': -13.034, 'SnLi': -18.26, 'SrLi': -0.27699999999999997, 'TaLi': 49.708999999999996, 'TiLi': 34.854, 'VLi': 38.228, 'WLi': 52.198, 'YLi': 7.907, 'YbLi': -0.6759999999999999, 'ZnLi': -6.893, 'ZrLi': 27.840999999999998, 'MnMg': 8.038, 'MoMg': 34.080999999999996, 'NbMg': 29.51, 'NdMg': -6.0, 'NiMg': -4.0, 'PdMg': -40.0, 'ScMg': -5.023, 'SiMg': -9.029, 'SnMg': -9.0, 'SrMg': -1.419, 'TaMg': 24.496, 'TiMg': 16.0, 'VMg': 21.11, 'WMg': 36.334, 'YMg': -6.0, 'YbMg': -6.292999999999999, 'ZnMg': -4.0, 'ZrMg': 3.228, 'MoMn': 4.945, 'NbMn': -4.0, 'NdMn': -0.348, 'NiMn': -8.0, 'PdMn': -23.0, 'ScMn': -8.363999999999999, 'SiMn': -45.0, 'SnMn': -18.439, 'SrMn': 18.601, 'TaMn': -3.8, 'TiMn': -8.094, 'VMn': -0.708, 'WMn': 6.2989999999999995, 'YMn': -1.4340000000000002, 'YbMn': 18.299, 'ZnMn': -13.345999999999998, 'ZrMn': -15.0, 'NbMo': -6.0, 'NdMo': 26.029, 'NiMo': -7.0, 'PdMo': -14.457, 'ScMo': 10.507, 'SiMo': -35.0, 'SnMo': 7.337999999999999, 'SrMo': 59.291000000000004, 'TaMo': -4.8580000000000005, 'TiMo': -3.5069999999999997, 'VMo': 0.01, 'WMo': -0.221, 'YMo': 24.004, 'YbMo': 54.835, 'ZnMo': 4.2330000000000005, 'ZrMo': -6.0, 'NdNb': 31.11, 'NiNb': -30.0, 'PdNb': -53.0, 'ScNb': 17.614, 'SiNb': -56.0, 'SnNb': -1.0, 'SrNb': 66.48, 'TaNb': 0.0, 'TiNb': 2.0, 'VNb': -1.0, 'WNb': -8.0, 'YNb': 29.302, 'YbNb': 61.126000000000005, 'ZnNb': -9.193999999999999, 'ZrNb': 4.0, 'NiNd': -30.0, 'PdNd': -81.59899999999999, 'ScNd': 1.238, 'SiNd': -64.374, 'SnNd': -62.01, 'SrNd': 8.696, 'TaNd': 28.69, 'TiNd': 16.271, 'VNd': 17.975, 'WNd': 25.796, 'YNd': 0.022000000000000002, 'YbNd': 9.363999999999999, 'ZnNd': -35.958, 'ZrNd': 10.269, 'PdNi': 0.0, 'ScNi': -37.938, 'SiNi': -40.0, 'SnNi': -15.136, 'SrNi': -7.983, 'TaNi': -29.0, 'TiNi': -35.0, 'VNi': -18.0, 'WNi': -3.0, 'YNi': -31.0, 'YbNi': -7.341, 'ZnNi': -15.802, 'ZrNi': -49.0, 'ScPd': -85.48899999999999, 'SiPd': -55.0, 'SnPd': -46.083999999999996, 'SrPd': -67.265, 'TaPd': -52.0, 'TiPd': -65.0, 'VPd': -35.111, 'WPd': -6.433, 'YPd': -82.235, 'YbPd': -61.787, 'ZnPd': -40.86, 'ZrPd': -91.0, 'SiSc': -64.891, 'SnSc': -54.997, 'SrSc': 17.991, 'TaSc': 15.706, 'TiSc': 7.4079999999999995, 'VSc': 7.037000000000001, 'WSc': 9.137, 'YSc': 0.934, 'YbSc': 16.421, 'ZnSc': -34.21, 'ZrSc': 4.0, 'SnSi': -11.0, 'SrSi': -20.004, 'TaSi': -56.0, 'TiSi': -66.0, 'VSi': -48.0, 'WSi': -31.0, 'YSi': -64.58800000000001, 'YbSi': -36.903, 'ZnSi': -0.845, 'ZrSi': -84.0, 'SrSn': -32.944, 'TaSn': -16.180999999999997, 'TiSn': -34.341, 'VSn': -13.04, 'WSn': 13.159, 'YSn': -61.261, 'YbSn': -48.28, 'ZnSn': 1.0, 'ZrSn': -43.0, 'TaSr': 62.966, 'TiSr': 44.06100000000001, 'VSr': 44.805, 'WSr': 60.364, 'YSr': 9.86, 'YbSr': -2.8310000000000004, 'ZnSr': -11.969000000000001, 'ZrSr': 37.849000000000004, 'TiTa': 1.0, 'VTa': -1.0070000000000001, 'WTa': -7.289, 'YTa': 26.945, 'YbTa': 57.99100000000001, 'ZnTa': -10.401, 'ZrTa': 3.0, 'VTi': -2.0, 'WTi': -5.636, 'YTi': 15.03, 'YbTi': 41.121, 'ZnTi': -22.324, 'ZrTi': 0.0, 'WV': -0.7929999999999999, 'YV': 16.512, 'YbV': 41.961999999999996, 'ZnV': -8.834, 'ZrV': -4.0, 'YW': 23.631999999999998, 'YbW': 55.691, 'ZnW': 7.684, 'ZrW': -9.0, 'YbY': 10.22, 'ZnY': -35.84, 'ZrY': 9.0, 'ZnYb': -22.791, 'ZrYb': 35.4, 'ZrZn': -36.735}

        AB = []
        for i in range(len(HEA)):
            for j in range(i, len(HEA)-1):
                AB.append(HEA[i] + HEA[j+1])

        del_Hab =[]
        for a in AB:
            if a in H_ab_1:
                del_Hab.append(H_ab_1.get(a))
            else:
                del_Hab.append(H_ab_2.get(a))
            
        omega = np.multiply(del_Hab, 4)

        C_i_C_j = []
        for i in range(len(HEA_)):
            for j in range(i, len(HEA_)-1):
                C_i_C_j.append(mole_fraction[i]*mole_fraction[j+1])

        del_Hmix = sum(np.multiply(omega, C_i_C_j))

        attributes.append(del_Hmix)

        R = 8.314

        del_Smix = -R*sum(np.multiply(mole_fraction, np.log(mole_fraction)))

        attributes.append(del_Smix)

        melting_temperature = {'Ag': 1234.0, 'Al': 933.25, 'Au': 1337.58, 'B': 2573.0, 'Be': 1551.0, 'Ca': 1112.0, 'Ce': 1071.0, 'Co': 1768.0, 'Cr': 2130.0, 'Cu': 1357.6, 'Fe': 1808.0, 'Gd': 1585.0, 'Hf': 2500.0, 'La': 1193.0, 'Li': 453.7, 'Mg': 922.0, 'Mn': 1517.0, 'Mo': 2890.0, 'Nb': 2741.0, 'Nd': 1289.0, 'Ni': 1726.0, 'Pd': 1825.0, 'Sc': 1812.0, 'Si': 1683.0, 'Sn': 505.06, 'Sr': 1042.0, 'Ta': 3269.0, 'Ti': 1933.0, 'V': 2175.0, 'W': 3680.0, 'Y': 1799.0, 'Yb': 1097.0, 'Zn': 692.73, 'Zr': 2125.0}

        Tm_i =[]
        for a in HEA:
            if a in melting_temperature:
                Tm_i.append(melting_temperature.get(a))
                
        Tm = sum(np.multiply(mole_fraction, Tm_i))

        Omega = (Tm*del_Smix)/abs(del_Hmix*1000)         # Converting Kilo Joules of del_Hmix to joules 

        attributes.append(Omega)

        Chi = {'Ag': 1.93, 'Al': 1.61, 'Au': 2.54, 'B': 2.04, 'Be': 1.57, 'Ca': 1.0, 'Ce': 1.12, 'Co': 1.88, 'Cr': 1.66, 'Cu': 1.9, 'Fe': 1.83, 'Gd': 1.2, 'Hf': 1.3, 'La': 1.1, 'Li': 0.98, 'Mg': 1.31, 'Mn': 1.55, 'Mo': 2.16, 'Nb': 1.6, 'Nd': 1.14, 'Ni': 1.91, 'Pd': 2.2, 'Sc': 1.36, 'Si': 1.9, 'Sn': 1.96, 'Sr': 0.95, 'Ta': 1.5, 'Ti': 1.54, 'V': 1.63, 'W': 2.36, 'Y': 1.22, 'Yb': 1.1, 'Zn': 1.65, 'Zr': 1.33}

        X_i =[]
        for a in HEA:
            if a in Chi:
                X_i.append(Chi.get(a))
                
        X_bar = sum(np.multiply(mole_fraction, X_i))

        del_Chi = (sum(np.multiply(mole_fraction, (np.subtract(X_i, X_bar))**2)))**0.5

        attributes.append(del_Chi)

        VEC_elements = {'Ag': 11, 'Al': 3, 'Au': 11, 'B': 3, 'Be': 2, 'Ca': 2, 'Ce': 3, 'Co': 9, 'Cr': 6, 'Cu': 11, 'Fe': 8, 'Gd': 3, 'Hf': 4, 'La': 3, 'Li': 1, 'Mg': 2, 'Mn': 7, 'Mo': 6, 'Nb': 5, 'Nd': 3, 'Ni': 10, 'Pd': 10, 'Sc': 3, 'Si': 4, 'Sn': 4, 'Sr': 2, 'Ta': 5, 'Ti': 4, 'V': 5, 'W': 6, 'Y': 3, 'Yb': 3, 'Zn': 12, 'Zr': 4}

        VEC_i =[]
        for a in HEA:
            if a in VEC_elements:
                VEC_i.append(VEC_elements.get(a))

        VEC = sum(np.multiply(mole_fraction, VEC_i))

        attributes.append(VEC)

        column_names = ['Ag', 'Al', 'Au', 'B', 'Be', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Fe', 'Gd', 'Hf', 'La', 'Li', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'Pd', 'Sc', 'Si', 'Sn', 'Sr', 'Ta', 'Ti', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr', 'Atomic size diff (δ)', 'ΔHmix', 'ΔSmix', 'Omega (Ω)', 'Δχ', 'VEC']

        attributes = pd.DataFrame([attributes], columns= column_names)

        model_path = resource_path('model.h5')
        robust_std_path = resource_path('robust_standardization.pkl')
        scaler_path = resource_path('scaler_standardization.pkl')

        saved_model = load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU(), 'ReLU': ReLU(), 'Softmax': Softmax()}, compile =False)

        input_attributes = 'Y' if int(saved_model.get_config().get('layers')[0].get('config').get('batch_input_shape')[1]) ==40 else 'N'

        if input_attributes == 'N':
            robust_1 = load(open(robust_std_path,'rb'))
            properties = attributes.drop(['Ag', 'Al', 'Au', 'B', 'Be', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Fe', 'Gd', 'Hf', 'La', 'Li', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'Pd', 'Sc', 'Si', 'Sn', 'Sr', 'Ta', 'Ti', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr'], axis = 1)
            pro = robust_1.transform(properties)
            attributes = pd.DataFrame(pro, columns=properties.columns)
        else:
            scaler_1 = load(open(scaler_path,'rb'))
            robust_1 = load(open(robust_std_path,'rb'))
            components = attributes.drop([ 'Atomic size diff (δ)', 'ΔHmix', 'ΔSmix', 'Omega (Ω)', 'Δχ', 'VEC'], axis = 1)
            properties = attributes.drop([ 'Ag', 'Al', 'Au', 'B', 'Be', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Fe', 'Gd', 'Hf', 'La', 'Li', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'Pd', 'Sc', 'Si', 'Sn', 'Sr', 'Ta', 'Ti', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr'], axis = 1)
            com = scaler_1.transform(components)
            pro = robust_1.transform(properties)
            comps = pd.DataFrame(com, columns=components.columns)
            props = pd.DataFrame(pro, columns=properties.columns)
            attributes = pd.concat([comps, props], axis=1)

        pred_phase = np.round(saved_model.predict(attributes),0)
        pp = pd.DataFrame(pred_phase, columns= ['AM','IM','SS','BCC1','FCC1','BCC2','FCC2'])

        phase_predicted =[]
        for a in pp:
            if pp[a][0]==1:
                phase_predicted.append(a)

        Phase =  ' + '.join(phase_predicted)
        if Phase == '':
            Phase = 'SORRY COULDNOT PREDICT ANY PHASES'


        t1 = Label(window, text="%s%s"%(round(atomic_size_difference*100,2), ' %'), anchor='center', width = 25, font='Helvetica 8 bold')
        t1.grid(row=2, column=3)
        t2 = Label(window, text="%s%s"%(round(del_Hmix,2),' kJ/mol'), anchor='center', width = 25, font='Helvetica 8 bold')
        t2.grid(row=3, column=3)
        t3 = Label(window, text="%s%s"%(round(del_Smix,2),' J/K/mol'), anchor='center', width = 25, font='Helvetica 8 bold')
        t3.grid(row=4, column=3)

        t0 = Label(window,text='%s%s'%(round(Tm,2), ' K'), anchor='center', width = 25, font='Helvetica 8 bold')
        t0.grid(row=5, column=3)

        t4 = Label(window, text="%s"%(round(Omega,2)), anchor='center', width = 25, font='Helvetica 8 bold')
        t4.grid(row=6, column=3)
        t5 = Label(window, text="%s"%(round(del_Chi,2)), anchor='center', width = 25, font='Helvetica 8 bold')
        t5.grid(row=7, column=3)
        t6 = Label(window, text="%s"%(round(VEC,2)), anchor='center', width = 25, font='Helvetica 8 bold')
        t6.grid(row=8, column=3)
        t7.delete("1.0", END)
        t7.insert(END, full_hea)
        t7.tag_configure("center", justify='center')
        t7.tag_add("center", 1.0, "end")
        t8.delete("1.0", END)
        t8.insert(END, Phase)
        t8.tag_configure("center", justify='center')
        t8.tag_add("center", 1.0, "end")


    b1 = Button(window, text="Predict Phase", command=properties, width = 15, font='Helvetica 12 bold') 
    b1.grid(row=0, column=2, columnspan=1)

    b2 = Button(window, text='RESTART', command = refresh, width = 15, font='Helvetica 12 bold')
    b2.grid(row =0, column = 3)

    p1 = Label(window, text='Atomic Size Difference (𝛿)', anchor='w', width = 25, font='Helvetica 8 bold')
    p1.grid(row=2, column=2)

    p2 = Label(window, text='Mixing of Enthalpy (ΔHmix)', anchor='w', width = 25, font='Helvetica 8 bold')
    p2.grid(row=3, column=2)

    p3 = Label(window, text='Mixing of Entropy (ΔSmix)', anchor='w', width = 25, font='Helvetica 8 bold')
    p3.grid(row=4, column=2)

    p0 = Label(window, text='Melting Tempt (Tm)', anchor='w', width = 25, font='Helvetica 8 bold')
    p0.grid(row=5, column=2)

    p4 = Label(window, text='Omega (Ω) = Tm*ΔSmix/|ΔHmix|', anchor="w", width = 25, font='Helvetica 8 bold')
    p4.grid(row=6, column=2)

    p5 = Label(window, text='Electronegativity (Δχ)', anchor='w', width = 25, font='Helvetica 8 bold')
    p5.grid(row=7, column=2)

    p6 = Label(window, text='VEC', anchor='w', width = 25, font='Helvetica 8 bold')
    p6.grid(row=8, column=2)

    p7 = Label(window, text='MPEA ENTERED', anchor='center', width = 50, font='Helvetica 10 bold')
    p7.grid(row=9, column=2, columnspan=2)

    t7 = Text(window, height=1, width=50, bg='white', fg='red')
    t7.configure(font=("Helvetica", 10, "bold"))
    t7.grid(row=10, column=2, columnspan=2)

    p8 = Label(window, text= 'PHASE PREDICTED', anchor='center', width=50, font='Helvetica 12 bold', fg='green')
    p8.grid(row=11, column=2, columnspan=2, rowspan=2)

    t8 = Text(window, height=1, width=25, bg='white', bd=10, fg='red', padx=10, pady=10)
    t8.configure(font=("Helvetica", 16, "bold"))
    t8.grid(row=13, column=2, columnspan=2, rowspan=2)


    window.mainloop()

if __name__ == '__main__':
    def refresh():
        window.destroy()
        start_gui()

    start_gui()