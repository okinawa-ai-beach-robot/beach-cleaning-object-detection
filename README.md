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
after this you should be able to start the notebooks like this:
```
cd [github repositor folder]
jupyter notebook
```
For kernel selection (in jupyter gui), you can select `beachbot` as python kernel.
*Dont* forget to always do a `conda activate beachbot` before starting `jupyter notebook` in the git root directory, as special commands like `% !pip install ...` will be executed in the terminal environment!

