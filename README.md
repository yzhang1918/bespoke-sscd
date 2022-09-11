# bespoke-sscd
A Re-implementation of BeSpoke (Bakshi et al. ICDM'18) for semi-supervised community detection.

- Faster Training due to the vectorized implementation
- Faster Evaluation

## Requirements

- python3
- numpy
- scipy
- scikit-learn
- tqdm

## Usage

```bash
python run.py --dataset dblp --train_size 100 --n_roles 4 --n_patterns 5 --pred_size 50000
```

Outputs:

```text
= = = = = = = = = = = = = = = = = = = =
##  Starting Time: 2019-12-05 14:00:32
[dblp] # Nodes: 317080 # TrainComms: 100 # TestComms: 4900
ComputeJaccards: 100%|███████████████████████████████████████████████████████| 317080/317080 [00:12<00:00, 25267.88it/s]
ExtractPercentiles: 100%|█████████████████████████████████████████████████████| 317080/317080 [00:33<00:00, 9365.59it/s]
NodeLocalFeature: 100%|██████████████████████████████████████████████████████| 317080/317080 [00:12<00:00, 25065.49it/s]
-> (All)  # Comms: 50000
AvgOverAxis0: Precision:0.56 Recall:0.32 F1:0.32 Jaccard:0.27
AvgOverAxis1: Precision:0.76 Recall:0.83 F1:0.75 Jaccard:0.68
AvgGlobal: Precision:0.66 Recall:0.57 F1:0.54 Jaccard:0.47
## Finishing Time: 2019-12-05 14:02:36
= = = = = = = = = = = = = = = = = = = =
```
(Extracting and evaluating 50K communities only took 2 minutes on my laptop!)

Check all available arguments:
```bash
python run.py --help
```

## References

1. Bakshi et al. 2018. Semi-Supervised Community Detection Using Structure and Size. ICDM.
1. The official implementation by the authors: [[repo]](https://github.com/abaxi/bespoke-icdm18).
