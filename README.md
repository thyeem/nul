<p align="center"> <img src="nul.png" height="200"/></p>

<!-- [![nul](https://img.shields.io/pypi/v/nul-lm)](https://pypi.org/project/nul-lm) -->

# nul
Using `nul`, anyone can easily create, train, and experiment with a customized language model from the
ground up. 


## Usage
```
$ nul -h
usage: nul [--conf] [-h] [-V]  ...

commands:
  
    new          create a model from a conf file
    ls           list models
    rm           remove models
    show         show information for a model
    train        train a model

options:
  --conf         show the default model configuration
  -h, --help     show this message
  -V, --version  show version information

```
Start with `nul --conf > pilot.conf` to create a default model configuration.  
Modify the parameter values, then generate a model with a name (e.g., `pilot`):

`nul new -f pilot.conf pilot`

Modify the training-related parameter values related to fit your preference as well.   
You can train the create model with your own data and tokenizer:

`nul train -f pilot.conf pilot`
