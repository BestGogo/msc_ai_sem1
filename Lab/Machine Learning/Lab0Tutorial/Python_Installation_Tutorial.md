For the first assignment of ECS708, we will use Python 3, along with its libraries NumPy and MatPlotLib. This is a simple document explaining how to prepare a Python environment that will support the above requirements.

Note: The guidelines mentioned below, have been tried in the PCs of ITL, using their CentOS operating system. So, make sure that you use CentOS in these PCs, by rebooting and selecting “CentOS” instead of Windows, if necessary.

Now, let's start!

Step 1: From the main menu, open a new terminal window, by searching for the keyword “terminal” in their app search menu.

Step 2: Download and install Anaconda, by typing the following commands:

```
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
bash Anaconda3-5.3.1-Linux-x86_64.sh
```

Press “ENTER” to continue, every time that you’ll be asked to do so. The installation may take up to 5 minutes. After finishing, you will see an **important** message, like
```
Installation finished.
Do you wish the installer to prepend the Anaconda3 install location
to PATH in your /home/*USERNAME*/.bashrc ? [yes|no]
```
Type “yes” and press “ENTER”. 

Note: If you fail to do so, you will not be able to use Anaconda by typing “conda” in your terminal. So make sure that you won’t skip this step, by typing anything else.

Step 3: Now you can use Anaconda to create a new Python environment, which you’ll use for Lab 1.

Type the following commands to create and activate a new environment, which you can name as “env_lab_1”:

```
conda create -n env_lab_1
source activate env_lab_1
```

*Optional step:*
You can navigate to the folder of that environment, and copy the files that you’re going to work with, there. Open a file manager, go to the home directory, and from there go to anaconda3/envs/env_lab_1. There, you can copy the files of Lab 1.

Step 4: You can execute the first and then the second script of Lab 1, by typing the following commands in a terminal.

</br>

```
cd path_to_your_ecs708_files/lab_1_part_1/1_one_variable
python ml_lab1.py
cd ../2_multiple_variables
python ml_lab2.py
```

*Optional for code editing:*
To edit your code, you may want to use PyCharm, which you can find from the app search menu of CentOS.

Open PyCharm, go to “File → New Project...”, give a name to your project, and click “Create”.

<div style="text-align:center"><img src="Create_Project.png" width="500"></div>
<!-- ![](./Create_Project.png =200x) -->

Go to “File→ Open...”, navigate to the directory of the code for Lab 1, select its folder, and click “OK”.

<div style="text-align:center"><img src="Open_folder.png" width="500"></div>
<!-- ![alt text](Open_folder.png) -->

In the left side of the window, you’ll now see your project’s folder structure:

<div style="text-align:center"><img src="Show_Project_Structure.png" width="500"></div>
<!-- ![alt text](Show_Project_Structure.png) -->

You can find the scripts that you want to edit in PyCharm from this structure, and open them by double-clicking. After modifying them, remember to save them, before trying to execute them from the terminal.
