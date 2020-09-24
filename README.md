# Search Data Structure Learning Experiments
The code for the experiments of Search Data Structure Learning. 

![Hamming Histogram](hamming_histogram.png?raw=true "Hamming Histogram")

![Halt SSWR Curves](halt_curves.png?raw=true "Halt SSWR Curves")

## How to Run

Calling `python run_<model type>.py --name=<name> --state_dir=<path> <optional parameters>` will starts (or resume) a training loop.
At each evaluation (each 500 batch) the model will be saved at `<state_dir>/current/<name>`. Also the top-5 models w.r.t the
Hamming Radius Search SSWR (halted@2081) and the top-5 models w.r.t. the Hashing Multi-Bernoulli Search SSWR (halted@5001) will be saved in
`<state_dir>/hr/<name>_hr<i>` and `<state_dir>/mb/<name>_mb<i>` with `<i>` in 0,1,2,3,4 (note that `<i>` is not ordering the models).

For the article we uses the following names `<model type>0`, ..., `<model type>4` for the non sharing models and `shared_<model type>0`, ..., `shared_<model type>4` for the sharing ones. To use the notebooks (SSWRCrunching and Visualization) it is preferable to use the same names, otherwise some modifications in the notebooks will be required.


### Requirements
* numpy
* scipy
* torch
* radbm

all accessible via `pip install <package>`

### Parameters
* --name=name -> the model name (used for saving states)
* --state_dir=path -> the path of the directory where the states need to be saved
* -h -> optional, for helps
* --nbits=n -> optional, provides the number of bits used in the models (n=64 by default)
* --share -> optional, for the query and document network to share parameters

### Examples
With 32 bits and without sharing, named fbeta1:
 `python run_fbeta.py fbeta1 /path/to/states_dir --nbits=32 --share`
 
With 64 bits (default) and with sharing, named shared_fbeta3:
 `python run_fbeta.py shared_fbeta3 /path/to/states_dir --share`