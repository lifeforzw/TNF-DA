
# Code Usage Guide

This guide provides a step-by-step explanation of how to use the provided code for preprocessing data and evaluating TNF-DA.



## 0. Path Preparation

To ensure the code runs smoothly, you first need to complete the path parameters in the code and a globals.yml, specifically the paths to the model and the data.

```python
datapth = ''
modelpth = ''
```

And then install the necessary dependencies using a `requirements.txt` file as

```python
pip install -r requirements.txt
```

## 1. Data Preprocessing

### Filtering Knowledge
To filter the knowledge that the model is certain about, use the `ifknow.py` script. This step ensures that only reliable knowledge is used for subsequent processing.

```bash
python ifknow.py
```

### Filtering Evaluation Data
After filtering, the `ifknoweval.py` script is used to generate the corresponding evaluation datasets. This step is crucial for preparing the data for the next stages.

```bash
python ifknoweval.py
```

## 2. Calculating Neuron IE Values

The `cma.py` script calculates the Information Entropy (IE) value for each neuron based on the knowledge. The results are stored in an NPZ file for later use.

```bash
python cma.py
```

## 3. Training with TNF-DA

To train the model using TNF-DA (Targeted Neuron Filtering with Data Augmentation), run the `kl_train.py` script. This script handles the entire training process, leveraging the preprocessed data and the calculated neuron IE values.

```bash
python kl_train.py
```

## 4. Ablation Testing without Data Augmentation

For ablation experiments where data augmentation is not used, utilize the `train.py` script. This allows for testing the performance of the model without the enhancement provided by data augmentation.

```bash
python train.py
```

 ## 5.  Summarize the Results

To summarize the results,  utilize the summary.py script. This helps to summarize the results and calculate the performance of editors. 

```bash
python summary.py
```

