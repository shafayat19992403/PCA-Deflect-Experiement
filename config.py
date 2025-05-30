import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

AGGR_MEAN = 'mean'
AGGR_GEO_MED = 'geom_median'
AGGR_FOOLSGOLD='foolsgold'
AGGR_PCA_DEFLECT = 'pca-deflect'
MAX_UPDATE_NORM = 1000  # reject all updates larger than this amount
patience_iter=20
AGGR_KRUM = 'krum'
AGGR_TRIMMED_MEAN = 'trimmed_mean'
AGGR_BULYAN = 'bulyan'
AGGR_CRFL='crfl'
AGGR_FEDAVGLR='fedavglr'
AGGR_MEDIAN='median'
AGGR_MKRUM = 'mkrum'
AGGR_FLTRUST='fltrust'
AGGR_FEDLDP='fedldp'
AGGR_FEDCDP='fedcdp'
AGGR_DNC = 'dnc'

TYPE_LOAN='loan'
TYPE_CIFAR='cifar'
TYPE_MNIST='mnist'
TYPE_FMNIST='fmnist'
TYPE_EMNIST='emnist'
TYPE_CIFAR100 = "cifar100"
TYPE_TINYIMAGENET='tiny-imagenet-200'