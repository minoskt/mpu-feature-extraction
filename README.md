# Mobile Phone Use Dataset: Feature Extraction Script

This script can be used as an example for extracting features from the [Mobile Phone Use Dataset](https://sites.google.com/view/mobile-phone-use-dataset) (also known as MPU). It is written in Python 3.7 and supports multi-processing (once core is used per user).

## Usage Information

First download the [MPU dataset](https://sites.google.com/view/mobile-phone-use-dataset) and decompress it under the same folder of the script.

The script uses some third-party libraries that need to be installed (eg. pandas, tqdm). Most of them are included with the [Anaconda 3.7 Distribution](https://www.anaconda.com/download/).

Execude the script using the command:

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

---
For more information about the dataset, please visit https://sites.google.com/view/mobile-phone-use-dataset.
