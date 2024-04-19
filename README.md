# Introduction
Tentative pyCATHY models for DA

# Installation
See [documentation](https://benjmy.github.io/pycathy_wrapper/content/installing.html)
Using **pygimli** I recommand to first install a conda env with: 
```
conda create -n pgCATHY -c gimli -c conda-forge "pygimli>=1.5.0"
conda activate pgCATHY
```
Then on top:
```
git clone https://github.com/BenjMy/pycathy_wrapper
cd pycathy_wrapper
python setup.py develop|install
```

gfortran and blas/lapack depedencies
```
sudo apt-get update
sudo apt-get install gfortran
sudo apt-get install blas-dev lapack-dev
```


# Notebooks

```
pip install jupyterlab
pip install jupyterlab_myst
```

- Testing
- Data Assimilation with point sensors
- Data Assimilation with ERT2D


# Authors
- B. Mary (ICA-CSIC)
