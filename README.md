# Mobile Phone Use Dataset: Feature Extraction Script

This script can be used as an example for extracting features from the [Mobile Phone Use Dataset](https://crawdad.org/telefonica/mobilephoneuse/20190429/) (also known as MPU dataset). It is written in Python 3.7 and supports multi-processing (one core is used per user). It uses 80% of your system cores by default, but can also be customised using [Optional Arguments](#Optional-Arguments).


## Usage Information

First download the [MPU dataset](https://sites.google.com/view/mobile-phone-use-dataset) and decompress it under the same folder of the script.

The script uses some third-party libraries that need to be installed (e.g. pandas and tqdm). Most of them are included with the [Anaconda 3.7 Distribution](https://www.anaconda.com/download/).

Execute the script using the command:

`
$ python feature_extraction.py
`


## Optional Arguments

```
  -h, --help            show this help message and exit
  -p nproc, --parallel nproc
                        execute in parallel, nproc=number of processors to
                        use.
  -sd [uuid [uuid ...]], --sudden-death [uuid [uuid ...]]
                        sudden death: use particular uuid to test the features
                        extraction; either specify the uuid or omit it and it
                        reads out a default one from code (ie. u000)
```
