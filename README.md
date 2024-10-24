# Machine Learning

Machine learning exercises

## Setup

Install [pyenv](https://github.com/pyenv-win/pyenv-win) and python `3.11` on **Windows**.

Install [tensorflow](https://www.tensorflow.org/install):

```
pip install --upgrade pip
pip install tensorflow
```

Install [keras](https://keras.io/getting_started/):

```
pip install --upgrade keras
```

Configure backend:

```
setx KERAS_BACKEND tensorflow /m
```

Install [jupyter](https://jupyter.org/install) server and notebook:

```
pip install jupyterlab
pip install notebook
```

Run jupyter:

```
jupyter lab
jupyter notebook
```

Install dependencies:

```
pip install pydot
pip install pydotplus
brew install graphviz
```

Install GPU support

https://schoolforengineering.com/tutorial/install-tensorflow-cuda-gpu-windows-10/