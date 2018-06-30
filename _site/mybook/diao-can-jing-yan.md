# 过拟合

To avoid overfitting, you can do a few things:

1) Use fewer parameters. Fewer layers, smaller layers.
2) Use more training material. Come up with more training data. Make sure it's meaningfully distinct. Perhaps save less for validation? (10%?)
3) Use response normalization, such as LRN layers. (This sometimes helps and sometimes not in my experience.)
4) Add dropout layers. Drop out at least 10% of data from the first perceptive layers; drop out at least 50% from the later decisive/analytic layers.

## Regulizer

weight_decay