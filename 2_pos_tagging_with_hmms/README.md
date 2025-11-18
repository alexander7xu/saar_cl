# Saar CoLi Assignment 2: POS tagging with HMMs

## Author

Juangui Xu

## Directory Structure

```
│   dataset.py                  -- Dataset-related functions and constants
│   main.ipynb                  -- Main assignment notebook
│   measure_multithreading.py   -- It will be executed in the Notebook
│   README.md                   -- This README file
│   test_ice_cream.py           -- Test the model with "ice-cream" problem
│   
└───hmm_pos_tagger              -- HMM POS Tagger classes
        base.py                 -- Interface and base classes
        concurrent_wrap.py      -- Multithreading support
        models.py               -- Implementation of models
        utils.py                -- Utilities, e.g. Counter class
        __init__.py
```

## Environments

### Device

- CPU: Snapdragon (TM) 8cx Gen 3 @ 2.69 GHz (8 Cores)
- RAM: 16 GB
- OS: Windows 11 Pro ARM 25H2

### Packages

To ensure the code produce correct results, you must use `python3.14t` or later version with free-threaded mode. Otherwise, GIL will limit the performance of multithreading. This will lead to a result that the speed decreases as the number of threads increases!

```
conllu==6.0.0
numpy==2.3.4
requests==2.32.5
torch==2.9.1
torchtyping==0.1.5
```

Because matplotlib does not have a wheel for Python3.14t on Windows, I have to use another virtual environments for plotting in notebook:

```
Python==3.13.9
conllu==6.0.0
ipykernel==7.0.1
matplotlib==3.10.7
requests==2.32.5
scikit-learn==1.7.2
torch==2.9.1
torchtyping==0.1.5
```

## Runtime

On my device, it requires:
- **4min** to test "K-Masked Augmentation" with different `K` varies from `1` to `50`.
- **4.5min** to test N-Gram HMM with different `N` varies from `1` to `10`.
- **2min** to test parallel performance with different `num_threads` varies from `0` to `10` with two different Tensor based implementation.
- **312min** for unsupervised training with 100 epochs. Take a Deep Sleep and wait for the result.
- **2min** for all the other tasks.

## Extra Points

- **Model IO:** `export` or `load` trained model.
- **Better unseen words handling:** I proposed "K-Masked Augmentation". The best model on `dev` dataset achieves 91.8500% accuracy on `test` dataset, better than 90.8007% by baseline.
- **N-Gram HMM:** Not only trigram, any positive integer `N` is acceptable.
- **Parallel Tagging:** Use multithreading to speed up tagging.
- **Tensor Based:** I've also implemented a special optimization, making it **~100% faster** in training.
- **Unsupervised Learning** and evaluate by `normalized_mutual_info_score`
