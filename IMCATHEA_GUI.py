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

from pymatgen.core.composition import Composition, Element
from matminer.featurizers.composition.alloy import Miedema, WenAlloys, YangSolidSolution
from matminer.featurizers.composition import ElementFraction
from matminer.featurizers.conversions import StrToComposition
from matminer.utils.data import MixingEnthalpy, DemlData
from matminer.utils import data_files #for Miedema.csv present inside package

ef = ElementFraction()
stc = StrToComposition()


def start_gui():
    global window
    window = tk.Tk()
    window.title("IMCATHEA")
    window.iconbitmap(resource_path('IMCATHEA.ico'))

    # For Title
    l1 = Label(window, text="No. of Components", font='Helvetica 8 bold')
    l1.grid(row=0, column=0)

    global HEA
    global HEA
    global HEA_
    global elements
    global thermo_values
    elements = ['Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ge', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'La', 'Ce', 'Nd', 'Gd', 'Yb', 'Hf', 'Ta', 'W', 'Au', 'Bi']
    thermo_values = ['Atomic size diff (δ)', 'ΔHmix', 'ΔSmix', 'Omega (Ω)', 'Δχ', 'VEC']

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
            elements2 = [n for n in sorted(elements) if n != HEA1.get()]
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
            HEA1= ttk.Combobox(window, value=sorted(elements), width=3, state='readonly')
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
            if (HEA_[a] == '1' or HEA_[a] == '1.0'):
                full.append(HEA[a])
            else:
                full.append(HEA[a]), full.append(HEA_[a])
        mpea = ''.join(full)

        # elem_prop_data = pd.read_csv(os.path.dirname(data_files.__file__) +"\\Miedema.csv", na_filter = False) #for Miedema.csv present inside package
        elem_prop_data = pd.read_csv(os.path.dirname(data_files.__file__) +"/Miedema.csv", na_filter = False) #for Miedema.csv present inside package
        VEC_elements = elem_prop_data.set_index('element')['valence_electrons'].to_dict()

        mole_fraction = []
        X_i = []
        r_i = []
        Tm_i = []
        VEC_i =[]
        R = 8.314
        for i in HEA:
            mole_fraction.append(Composition(mpea).get_atomic_fraction(i))
            X_i.append(Element(i).X)
            r_i.append(Element(i).atomic_radius) if Element(i).atomic_radius_calculated == None else r_i.append(Element(i).atomic_radius_calculated)
            Tm_i.append(Element(i).melting_point)
            try: VEC_i.append(DemlData().get_elemental_property(Element(i), "valence"))
            except KeyError:
                if i in VEC_elements: VEC_i.append(float(VEC_elements.get(i)))

        HEA_ = [float(i) for i in HEA_]
        total_mole = sum(HEA_)
        composition = dict.fromkeys(elements, 0)

        for i in range(len(HEA)):
            if HEA[i] in composition:
                composition.update({HEA[i]:mole_fraction[i]})
        composition_list = list(composition.values())
        attributes = list(composition.values())

        # Atomic Size Difference Calculation
        r_bar = sum(np.multiply(mole_fraction, r_i))
        term = (1-np.divide(r_i, r_bar))**2
        atomic_size_difference = sum(np.multiply(mole_fraction, term))**0.5

        attributes.append(atomic_size_difference)

        # Enthalpy of Mixing Calculation
        AB = []
        C_i_C_j = []
        del_Hab = []
        for i in range(len(HEA)):
            for j in range(i, len(HEA)-1):
                AB.append(HEA[i] + HEA[j+1])
                C_i_C_j.append(mole_fraction[i]*mole_fraction[j+1])
                del_Hab.append(round(Miedema().deltaH_chem([HEA[i], HEA[j+1]], [0.5, 0.5], 'ss'),3))
        #         del_Hab.append(MixingEnthalpy().get_mixing_enthalpy(Element(HEA[i]), Element(HEA[j+1]))) # Matminer MixingOfEnthalpy
        omega = np.multiply(del_Hab, 4)
        del_Hmix = sum(np.multiply(omega, C_i_C_j))

        attributes.append(del_Hmix)

        # Entropy of Mixing Calculation
        del_Smix = -WenAlloys().compute_configuration_entropy(mole_fraction)*1000
        # del_Smix = -R*sum(np.multiply(mole_fraction, np.log(mole_fraction)))

        attributes.append(del_Smix)


        # Average Melting Temperature Calculation        
        Tm = sum(np.multiply(mole_fraction, Tm_i))

        # Omega parameter Calculation
        Omega = (Tm*del_Smix)/abs(del_Hmix*1000)         # Converting Kilo Joules of del_Hmix to joules 

        attributes.append(Omega)

        # Electronegativity Calculation                
        X_bar = sum(np.multiply(mole_fraction, X_i))
        del_Chi = (sum(np.multiply(mole_fraction, (np.subtract(X_i, X_bar))**2)))**0.5

        attributes.append(del_Chi)

        # Valence Electron Concentration Calculation
        VEC = sum(np.multiply(mole_fraction, VEC_i))

        attributes.append(VEC)

        column_names = elements + thermo_values 

        attributes = pd.DataFrame([attributes], columns= column_names)

        model_path = resource_path('model.h5')
        robust_std_path = resource_path('robust_standardization.pkl')
        scaler_path = resource_path('scaler_standardization.pkl')

        saved_model = load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU(), 'ReLU': ReLU(), 'Softmax': Softmax()}, compile =False)

        input_attributes = 'Y' if int(saved_model.get_config().get('layers')[0].get('config').get('batch_input_shape')[1]) ==47 else 'N'

        if input_attributes == 'N':
            robust_1 = load(open(robust_std_path,'rb'))
            properties = attributes[thermo_values]
            pro = robust_1.transform(properties)
            attributes = pd.DataFrame(pro, columns=properties.columns)
        else:
            scaler_1 = load(open(scaler_path,'rb'))
            robust_1 = load(open(robust_std_path,'rb'))
            components = attributes[elements]
            properties = attributes[thermo_values]
            com = scaler_1.transform(components)
            pro = robust_1.transform(properties)
            comps = pd.DataFrame(com, columns=components.columns)
            props = pd.DataFrame(pro, columns=properties.columns)
            attributes = pd.concat([comps, props], axis=1)

        pred_phase = np.round(saved_model.predict(attributes),0)
        pp = pd.DataFrame(pred_phase, columns= ['AM','IM','SS','BCC1','FCC1','BCC2','FCC2'])

        # print(attributes["Bi"])
        phase_predicted =[]
        for a in pp:
            if pp[a][0]==1:
                phase_predicted.append(a)

        if 'IM' in phase_predicted:
            Phase = "System has IMC"
        else:
            Phase = "System doesn't have IMC"

        # Phase =  ' + '.join(phase_predicted)
        # if Phase == '':
        #     Phase = 'SORRY COULDNOT PREDICT ANY PHASES'


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
        t7.insert(END, mpea)
        t7.tag_configure("center", justify='center')
        t7.tag_add("center", 1.0, "end")
        t8.delete("1.0", END)
        t8.insert(END, Phase)
        t8.tag_configure("center", justify='center')
        t8.tag_add("center", 1.0, "end")


    b1 = Button(window, text="Detect IMC", command=properties, width = 15, font='Helvetica 12 bold') 
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

    p8 = Label(window, text= 'IMC Presence or Absence', anchor='center', width=50, font='Helvetica 12 bold', fg='green')
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