# pyMPEALab

pyMPEALab is a python-based Multi-Principle Elements Alloy Laboratory software capable of predicting the phases of Multi-Principle Elements Alloy using Neural Network Algorithm.


## For Linux User:
Python with following libraries are needed to run the GUI application: Tensorflow, Pandas, Numpy, Pickle, Sklearn, and tkinter
pyMPEALab also needs 4 extra supporitng files (1 model file, 2 standardization files, and a icon file) to operate, which will be made available from the author upon reasonable request. 


## For Windows User:
User can run "pyMPEALAB_GUI.exe" wihout any softwares to be installed.


## Procedure for using "pyMPEALab" GUI Application:

Step 0. Open / Run "pyMPEALab_GUI.py" (for Linux users) & "pyMPEALab_GUI.exe" (for Windows users) file to open the GUI application

Step 1. User need to select the No. of Elements/Components in the MPEA whose phase is to be predicted, from the dropdown menu at the top left corner.
		    After selection of element size/number (from 2 to 10),
		    
Step 2. User can select each element of MPEA, one at a time from the drop down menu generated just below "No. of Component" tab

Step 3. After selection of each element a blank space is provided just at the right side of selected element tab where user need to enter the corresponding composition/elemental           fraction of the element.

Step 4. Repeat Step 2-3 until the last element of the MPEA and it's composition/elemental fraction is entered.

Step 5. Press "Predict Phase" tab to get the prediction along with the physical properties of the entered MPEA displayed in the right side of the GUI application.

Step 6. If user wants to predict the phase for another MPEA, user can click on "Restart" tab at the right top side of GUI application to restart the application instantly.


If the user makes errors while selecting any options from dropdown menu as suggested in Step 1,2,3,4 user can click on "Restart" tab at the right top side of GUI application to restart the application instantly and start the process for prediction again from Step 1.



## Video (YouTube) Tutorial for using pyMPEALab:
The stepwise process to predict the phase for an example case of "Cu0.5NiCoCrAl0.5Fe3.5" MPEA is shown in the YouTube video tutorial at: https://youtu.be/pijwp6rXyYQ



## How to cite pyMPEALab libraries:
If you use pyMPEALab in your research, please cite:

"U. Subedi, A. Kunwar , Y.A. Coutinho, K. Gyanwali . pyMPEALab toolkit for accelerating phase design in multi-principal element alloys. Metals  and Materials International vol, pppp-pppp (2021-2022)."
