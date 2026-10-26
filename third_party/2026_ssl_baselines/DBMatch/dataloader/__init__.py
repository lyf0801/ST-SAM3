import numpy as np
from torch.utils import data
from dataloader.dataset_binary import RS_Binary_DataSet
# from dataloader.dataset_multi import VOC_DataSet



def CreateTrainDataLoader(args):
    train_l_dataset = RS_Binary_DataSet(args.data_dir, args.dataset, args.list_train_l, args.algorithm, \
                                        set_mode='labeled_training', ignore_index=args.ignore_index)
    if args.algorithm == 'UniMatch':
        train_u_dataset = RS_Binary_DataSet(args.data_dir, args.dataset, args.list_train_u, args.algorithm, \
                                            set_mode='unlabeled_training_multi', ignore_index=args.ignore_index)
    else:
        train_u_dataset = RS_Binary_DataSet(args.data_dir, args.dataset, args.list_train_u, args.algorithm, \
                                            set_mode='unlabeled_training', ignore_index=args.ignore_index)
    val_dataset = RS_Binary_DataSet(args.data_dir, args.dataset, args.list_val, args.algorithm, \
                                    set_mode='test', ignore_index=args.ignore_index)

    train_l_dataloader = data.DataLoader(train_l_dataset,
                                         batch_size=args.batch_size, 
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         drop_last=False,
                                         pin_memory=False)
    train_u_dataloader = data.DataLoader(train_u_dataset,
                                         batch_size=args.batch_size, 
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         drop_last=True,
                                         pin_memory=False)
    val_dataloader = data.DataLoader(val_dataset,
                                     batch_size=args.batch_size*2, 
                                     shuffle=False, 
                                     num_workers=args.num_workers, 
                                     drop_last=False,
                                     pin_memory=False)
    return train_l_dataloader, train_u_dataloader, val_dataloader


def CreateTestDataLoader(args):
    if 'INRIA' in args.dataset or 'Building' in args.dataset or 'Road' in args.dataset:
        test_dataset = RS_Binary_DataSet(args.data_dir, args.dataset, args.list_test, args.algorithm, \
                                        set_mode='test', ignore_index=args.ignore_index)
    else:
        raise ValueError('The dataset is out of range')
    test_dataloader = data.DataLoader(test_dataset,
                                      batch_size=args.batch_size, 
                                      shuffle=False, 
                                      num_workers=args.num_workers, 
                                      drop_last=False,
                                      pin_memory=False)
    return test_dataloader




