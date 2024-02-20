# beach-cleaning-object-detection
Object detection model, and training data for use in a beach cleaning robot based in Okinawa, Japan


# Install
## Notebooks
In case you want to use conda virtual environment management,
installation can  be done as follows:
```
conda create --name beachbot
conda activate beachbot

pip install notebook *[or alternatively pip install jupyterlab]*
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=beachbot
```
Alternatively, jupyter notebook can be installed via conda:
```
conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
```

After this you should be able to start the notebooks like this:
```
cd [github repositor folder]
jupyter notebook
```
For kernel selection (in jupyter gui), you can select `beachbot` as python kernel.
*Dont* forget to always do a `conda activate beachbot` before starting `jupyter notebook` in the git root directory, as special commands like `% !pip install ...` will be executed in the terminal environment!

## Learned Models:
The learned models and model exports as ONNX, tensorflow & TensorRT are stored for now in Dropbox.
To use them you have to:
1. Download this zip into the **_Models_** folder: https://www.dropbox.com/scl/fi/met65imjnito6x7wowe1g/beachbot_yolov5s_beach-cleaning-object-detection__v1i__yolov5pytorch.zip?rlkey=0vb4yt8ofppb08vj7imsp0vxc&dl=0
2. Extract the archive: `unzip beachbot_yolov5s_beach-cleaning-object-detection__v1i__yolov5pytorch.zip`


## Datasets:
Download/export zip files of dataset `beach-cleaning-object-detection.v1i.yolov5pytorch` from Roboflow into **_Datasets_** folder