# Data folder

## creditcard.csv (NOT included in this repository)

Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and place it at `data/creditcard.csv`.

- Source: Worldline and the ULB Machine Learning Group
- License: Database Contents License (DbCL) v1.0
- 284,807 transactions over two days in September 2013, 492 frauds (0.172%)

### Columns
- `Time`: seconds since the first transaction (not a date)
- `V1`-`V28`: Anonymized features. All of them were transformed with PCA for confidentiality.
- `Amount`: transaction amount
- `Class`: 1 = fraud, 0 = genuine

### Limitations
- No merchant name
- It is easy to achieve 98.83% accuracy my marking all the transactions as not fraud but that is not what we expect here so accuracy is a bad metric

### Citation
Dal Pozzolo et al., "Credit card fraud detection: a realistic modeling and a novel learning strategy", IEEE Transactions on Neural Networks and Learning Systems, 2018.

## sample_transactions.csv
Synthetic (simulated) demo data. Will be generated in a later step.