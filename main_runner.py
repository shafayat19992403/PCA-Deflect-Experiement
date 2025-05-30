import argparse


parser = argparse.ArgumentParser(description='PPDL')
parser.add_argument('--attack', dest='attack', required=True,
                    choices=['dba', 'cas', 'baseline'],
                    help='Type of attack to run')
parser.add_argument('--defense', dest='defense', required=True, choices=['mean', 'pca-deflect'], help='Type of defense to run')
parser.add_argument('--dataset', dest='dataset', choices=['cifar,mnist,fmnist,emnist'])
args = parser.parse_args()
if args.attack == 'dba':
    import attacks.DBA.main as main
    main.main_dba(args.defense, args.dataset)