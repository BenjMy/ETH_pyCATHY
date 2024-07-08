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

- Testing (see pyCATHY doc)
- Data Assimilation with point TDR sensors PARTI: [here](./notebooks/DA_SMC_sensors_part1.ipynb)
- Data Assimilation with point TDR sensors PARTII: [here](./notebooks/DA_SMC_sensors_part2.ipynb)
- Data Assimilation with ERT2D (in progress)


# Authors
- B. Mary (ICA-CSIC)
