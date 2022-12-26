<p align="center">
  <img width="500" src="https://cdn.discordapp.com/attachments/1055995879292666009/1055996316607598683/Diff-SVC_Logo.png">
  </p>
<p align="center">
  <img width="100" src="https://cdn.discordapp.com/attachments/1055995879292666009/1055996316351742054/Diff-SVC_small.png">
  <h1 align="center" style="margin: 0 auto 0 auto;">DIFF-SVC GUI</h1>
  <h5 align="center" style="margin: 0 auto 0 auto;">Interface for DIFF-SVC</h5>
  </p>

**Diff-SVC GUI** is a program that only works **attached to**: https://github.com/prophesier/diff-svc

Diff-SVC is a _Singing Voice Conversion via diffusion model_

Warning: This project is in development and right now it only has the GUI done for Rendering purposes
_Made in Python 3.8.10_



## Deployment

To deploy this project you need to have the requeriments for [diff-svc](https://github.com/prophesier/diff-svc) installed.

_Guide in english for diff-svc and what it does [here](https://docs.google.com/document/d/1nA3PfQ-BooUpjCYErU-BHYvg2_NazAYJ0mvvmcjG40o/edit#heading=h.6q7sdk7zbgfj)_

You first need to install the requeriments for this project, run in your terminal:
```bash
  pip install tkinter
```
Once you have done that, get the [source files](https://github.com/Kangarroar/diff-svc-GUI/tree/main/Diff-SVC%20Code) and drag and drop them on your diff-svc main folder.
Then run a terminal and point it to your diff-svc folder and write
```bash
  python DIFFSVCGUI.py
  #In case that this does not work write:
  python3 DIFFSVCGUI.py
```
And the program will load.



## Features

- Intuitive interface instead of depending of full coding
- Modifiable parameters like
    - KEY, WAV, SPEEDUP, Gender Flag, Noisestep, Threshold.
_More information about those parameters here in the_ [DIFF-SVC Documentation](https://docs.google.com/document/d/1nA3PfQ-BooUpjCYErU-BHYvg2_NazAYJ0mvvmcjG40o/edit#heading=h.6q7sdk7zbgfj)


## Screenshots

![App Screenshot](https://i.ibb.co/0JQj9qj/DIFF-SVC.png)

<details>
  <summary>*Updates & Roadmap*</summary>

  ## Roadmap
Rightnow the GUI is fully done in Python and I was planning on making it like that until it works 100%, but now I am working on a better looking GUI here's a sneakpeak for now ;)

![App Screenshot](https://i.ibb.co/swzzZkb/placeholdertesting.png)

Having in mind to:
- Finish the Training Tab
- Add some cute sounds to the GUI to make it more alive
- Make it multithreading
</details>

