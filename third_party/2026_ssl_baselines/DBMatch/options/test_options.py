import argparse

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="DBMatch for Semi-supervised Building Footprint Extraction")
        # task settings
        parser.add_argument("-dataset", type=str, default='INRIA', help="used dataset", 
                                choices=['INRIA', 'INRIA/Austin', 'INRIA/Vienna', 'INRIA/Kitsap', 'INRIA/Tyrol', 'INRIA/Chicago', \
                                         'GF7_Buildings', 'GF7_Buildings/Chongqing', 'GF7_Buildings/Guangzhou', 'GF7_Buildings/Lanzhou', \
                                         'GF7_Buildings/Ningbo', 'GF7_Buildings/Shenzhen', 'GF7_Buildings/Tianjin', \
                                         'Massachusetts_Buildings'])
        parser.add_argument("-num-classes", type=int, default=1, help="Number of classes.")
        parser.add_argument("-ignore-index", type=int, default=-1, help="Ignored class index.")
        parser.add_argument("-percent", type=float, default=1, choices=[1, 2, 5, 10, 20, 100])

        # directory and file paths
        parser.add_argument("-data-dir", type=str, default='/home/Datasets/RSSeg')
        parser.add_argument("-pred-dir", type=str, default='./predicts')
        parser.add_argument("-list-test", type=str, default='lists/test.txt')
        parser.add_argument("-save-dir", type=str, default='./checkpoints', help="Where to save snapshots of the model.")

        # algorithm
        parser.add_argument("-algorithm", type=str, default='OnlySup', help="selected method for model training.", \
                                choices=['OnlySup', 'FullySup', 'CCT', 'CPS', 'CutMix', 'DebiasPL', \
                                         'FixMatch', 'FixMatch', 'UniMatch', 'AdaptMatch', 'DBMatch', 'NDBMatch'])

        # model training
        parser.add_argument("-model", type=str, default='Deeplab_V3plus', help="feature extraction models ", 
                                choices=['Deeplab_V3plus', 'SegFormer', 'HRNet'])
        parser.add_argument("-batch-size", type=int, default=8, help="input batch size.")
        parser.add_argument("-learning-rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
        parser.add_argument("-num-workers", type=int, default=4, help="number of threads.")
        return parser.parse_args()
    
    def print_options(self, args):
        message = ''
        message += '--------- Options --------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '--------- End ----------'
        print(message)

