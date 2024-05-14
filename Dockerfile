FROM condaforge/mambaforge

# Set working directory for the project
WORKDIR /app

# Create Conda environment from the YAML file
COPY environment.yml .
#RUN mamba env create -f environment.yml

# Override default shell and use bash
#SHELL ["conda", "run", "-n", "monai13", "/bin/bash", "-c"]

# Activate Conda environment and check if it is working properly
#RUN echo "Making sure PyTorch is installed correctly..."
#RUN conda run -n monai13 python -c "import torch"

# Python program to run in the container
COPY deepmitral.py .
COPY model/pretrained_model.pt model.md
ENTRYPOINT ["conda", "run", "-n", "monai10", "python", "deepmitral.py", "segment", "model.md", "input"]