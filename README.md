# MaQIP
Many Qsvms In Parallel

## Run Parallel QSVM on QPU example
Prepare a Python environment and activate it. (It should work with any Python version > 3.8). For example, through `venv`:

    python -m venv <environment_name>

### Clone the Repository

Clone the MaQIP repository and install the maqip package:

    git clone https://github.com/rmoretti9/MaQIP.git

    cd MaQIP
    pip install -e .

### Configure IBM Account

Inside `MaQIP/examples/torino_run.py` (right after the imports), fill the IBM account/provider fields with your token/instance as required and save the changes.

### Run the Example Script

<b> Move inside the examples folder </b> (this is important for the correct execution) and run the script. This will prepare the Quantum Circuits and send batch jobs to IBM Torino:

    cd examples
    python torino_run.py

This script will generate a folder called `Output_genetic` that will progressively save data as they are generated. Do not rename the `Output_genetic` folder and its content.