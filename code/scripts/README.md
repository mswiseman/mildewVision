1. **plot_sal_map_leaf.sh** is a script that runs does classification, saliency mapping, and disease severity rate calculation leaf disk inmages given a pretrained model.
2. **plot_sal_map_patch.sh** is a script that runs does classification, saliency mapping, and disease severity rate calculation on a 224x224x3 patch from a larger leaf disk image.
3.  **plot_sal_map_leaf_all.sh** is a script designed to run multiple leaf analyses in parallel. As written, there are examples of running `../plot_sal_map_leaf.py` and/or `../leaf_correlation.py`.
4. **train.sh** is used to train a classification model.
5. **train_seg.sh** is used to train a segmentation model.
