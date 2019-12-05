import argparse
import datetime

from bespoke import *

from utils import *


def evaluate(pred_comms, test_comms):
    scores_1, scores_2 = eval_comms_bidirectional(pred_comms, test_comms)
    mean_score_1 = scores_1.mean(0)
    print_results(mean_score_1, 'AvgOverAxis0')
    mean_score_2 = scores_2.mean(0)
    print_results(mean_score_2, 'AvgOverAxis1')
    mean_score_all = (mean_score_1 + mean_score_2) / 2.
    print_results(mean_score_all, 'AvgGlobal')
    return mean_score_1, mean_score_2, mean_score_all


def main(args):
    adj_mat, comms, *_ = load_snap_dataset(args.dataset, args.root)
    # Split comms
    train_comms, test_comms = split_comms(comms, args.train_size, args.seed, args.max_size)
    print(f'[{args.dataset}] # Nodes: {adj_mat.shape[0]}'
          f' # TrainComms: {len(train_comms)} # TestComms: {len(test_comms)}',
          flush=True)
    # Fit
    model = Bespoke(args.n_roles, args.n_patterns, args.eps, unique=True)
    model.fit(adj_mat, train_comms)
    pred_comms = model.sample_batch(args.pred_size)
    # Evaluating
    print(f'-> (All)  # Comms: {len(pred_comms)}')
    evaluate(pred_comms, test_comms)
    # Save
    if len(args.save_dst) > 0:
        write_comms_to_file(pred_comms, args.save_dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name', default='dblp')
    parser.add_argument('--root', type=str, help='data directory', default='datasets')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--train_size', type=int, help='the number of training communities', default=500)
    parser.add_argument('--max_size', type=int,
                        help='Communities whose size is larger than this value will be discarded.',
                        default=0)
    parser.add_argument('--n_roles', type=int, help='the number of node labels', default=4)
    parser.add_argument('--n_patterns', type=int, help='the number of community patterns', default=5)
    parser.add_argument('--eps', type=int, help='maximum tolerance for seed selection', default=5)
    parser.add_argument('--pred_size', type=int, help='the number of communities to extract', default=50000)
    parser.add_argument('--save_dst', type=str, help='where to save the searched communities',
                        default='bespoke_comms.txt')

    args = parser.parse_args()
    seed_all(args.seed)

    print('= ' * 20)
    print('##  Starting Time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    main(args)
    print('## Finishing Time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)
