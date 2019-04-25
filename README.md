# RTree Learned Index
This is the main code setup used to carry out Fall 2019 Research on Learned Indexes for R-Trees.


# Dependencies

  - Sklearn
  - Pytorch
  - Numpy
  - Pandas
  - pyrtree (needs to be installed)

# Running Scripts
All the training scripts are present within pyrtree/tests

## Building Dataset
Run buildset to build different types of datasets. Choices are present in argparse. Either path or LeafID can be produced. Rectangle distribution can be modified in test_rtree.

## Training Models
### Hybrid Model
The hybrid Rtree model can be trained in train_hybrid. Tree parameters and approximators can be modified within the script. 

### Neural Network
Direct Mapping using Neural Network can be carried out using the NN/ directory. This requires pytorch. Make sure to normalize the dataset before running. 

## Rtree Implementation Source
https://github.com/Rhoana/pyrtree
Double check the implementation (I had a couple issues pop up that I wasn't able to pursue.)
