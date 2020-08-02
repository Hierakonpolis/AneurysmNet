# Use the official image as a parent image.
FROM nvidia/cuda:10.2-base
#FROM anibali/pytorch:1.5.0-cuda10.2
FROM pytorch/pytorch

# Set the working directory.
WORKDIR /AneurysmNet

# Copy the file from your host to your current location.
# COPY package.json .

# Run the command inside your image filesystem.
# RUN conda install -c conda-forge scikit-image nibabel scipy
# RUN conda install numpy tqdm 
RUN pip install numpy tqdm scikit-image nibabel scipy

# ADD localfolder dockerfolder

# Copy the rest of your app's source code from your host to your image filesystem.
COPY . .

# Run the specified command within the container.
# CMD [ "python3", "inference.py" ]
