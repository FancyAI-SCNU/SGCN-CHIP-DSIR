import os
import argparse
import pickle
import argparse
import datetime

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model import TrajectoryModel

from tools.metrics import *
from loader import TrajectoryDataset
from tools.utils import *
from sinkhorn_sort import Rank_Sort_log
from path_utils import calc_binary_path_energy, nodes_rel_to_nodes_abs_tensor

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='number of epochs')
    parser.add_argument('--clip_grad', type=float, default=10,
                        help='gradient clipping')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum of lr')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight_decay on l2 reg')
    parser.add_argument('--lr_sh_rate', type=int, default=100,
                        help='number of steps to drop the lr')
    parser.add_argument('--milestones', type=int, default=[50, 100],
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=True,
                        help='Use lr rate scheduler')
    parser.add_argument('--tag', default='',
                        help='personal tag for the model ')
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('-p', '--model_dir', type=str, default='',
                        help='path of test models')
    parser.add_argument('-d', '--dataset', default='eth', required=True,
                        help='eth,hotel,univ,zara1,zara2')
    parser.add_argument('--eps', type=float, default=2e-3,
                        help='rank operation eps')
    parser.add_argument('--seed', type=int, default=1357,
                        help='random seed')
    parser.add_argument('--social_dist_sigma', type=int, default=1,
                        help='social distance, set to 2m')
    parser.add_argument('-uc', '--use_clip', type=int, default=1, choices=[0, 1],
                        help='use clip contrastive loss')
    parser.add_argument('-up', '--use_pairwise_rel', type=int, default=0, choices=[0, 1],
                        help='use future pairwise distance regression')

    args = parser.parse_args()

    print("Training initiating....")
    print(args)


    def graph_loss(V_pred, V_target):
        return bivariate_loss(V_pred, V_target)


    if len(args.tag) == 0:
        args.tag = args.dataset

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # region Parameters
    obs_seq_len = args.obs_len
    pred_seq_len = args.pred_len
    data_set = './datasets/' + args.dataset + '/'

    use_clip = args.use_clip
    # use_unary = args.use_unary
    # use_binary = args.use_binary
    use_pairwise_rel = args.use_pairwise_rel

    if use_clip > 0:
        print('Use CLIP !!!!')
    if use_pairwise_rel > 0:
        print('Use Pairwise relation !!!!')

    use_finetune = False
    if use_clip or use_pairwise_rel:
        # args.batch_size = args.batch_size // 2
        # args.num_epochs = 200
        # args.weight_decay = 0.0001
        args.lr *= 0.5
        print('Use new batch size')
        use_finetune = True

    if len(args.tag) == 0:
        args.tag = args.dataset
    # endregion

    # region SaveDir
    result_dir = './result/'
    print('Checkpoint dir %s' % (result_dir,))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    run_id = datetime.datetime.now().strftime('%m-%d-%H-%M')
    checkpoint_dir = os.path.join(result_dir, "%s_uc%d_up%d_%s" % (
        args.tag, use_clip, use_pairwise_rel, run_id))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print('Checkpoint dir %s' % (checkpoint_dir,))

    with open(os.path.join(checkpoint_dir, 'args.pkl'), 'wb') as fp:
        pickle.dump(args, fp)
    # endregion

    processed_path = './dataset_processed/'
    os.makedirs(processed_path, exist_ok=True)
    processed_data_set = './dataset_processed/' + args.dataset + '/'
    os.makedirs(processed_data_set, exist_ok=True)

    # region Dataset selection
    dset_train = TrajectoryDataset(
        data_set + 'train/',
        processed_data_set + 'train.pth',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_train = DataLoader(
        dset_train,
        batch_size=1,
        shuffle=True,
        num_workers=6)

    dset_val = TrajectoryDataset(
        data_set + 'val/',
        processed_data_set + 'val.pth',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_val = DataLoader(
        dset_val,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=4)
    # endregion

    # region model construction
    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999, 'min_train_epoch': -1,
                        'min_train_loss': 9999999999999999}
    print('Training started ...')

    model = TrajectoryModel(number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                            obs_len=8, pred_len=12, n_tcn=5, out_dims=5).to(device)

    # total = sum(p.numel() for p in model.parameters())
    # print("Total params: %.2fM" % (total / 1e6), total)
    # exit()

    print(args.model_dir)
    if len(args.model_dir) > 0:
        model_path = os.path.join(args.model_dir, 'val_best.pth')
        print(model_path)
        if os.path.isfile(model_path):
            # model.load_state_dict(torch.load(model_path))
            # print('Load %s' % (model_path,))
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_path), strict=False)
            else:
                model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        else:
            print(model_path,'not exists')
            assert 1 == 2

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    if args.use_lrschd:
        if use_finetune:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200, 500], gamma=0.6)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)
    # endregion

    # region Diff-Sort Rank_Sort_Log parameters
    p1 = torch.tensor(math.sqrt(3.0) / math.pi).float().to(device)
    min_std = torch.tensor(1e-10).float().to(device)
    two_sigma_square = 2 * args.social_dist_sigma ** 2


    # endregion

    def train(epoch, model, optimizer, checkpoint_dir, loader_train):
        global metrics, constant_metrics
        model.train()
        loss_batch = 0
        batch_count = 0
        is_fst_loss = True
        loader_len = len(loader_train)
        pred_seq_len = args.pred_len
        # turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1
        total_iter = loader_len // args.batch_size

        model.train()
        cur_iter = 0
        for cnt, batch in enumerate(loader_train):
            if cur_iter >= total_iter:
                print('break at batch %d/%d of total sample %d/%d' % (cur_iter, total_iter, batch_count, loader_len))
                assert batch_count + args.batch_size > loader_len
                break

            batch_count += 1

            # Get data
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, V_tr = batch

            # V_obs [1, 8, N_obj, 3(x,y, pos_enc)]
            N_obj = V_obs.size(2)
            assert V_obs.size(1) == obs_seq_len
            # V_obs size [1, 8, N_obj, 3]
            identity = get_ID(obs_seq_len, N_obj, device)

            optimizer.zero_grad()

            # [fut_len=12, N, dim=5], dim is bi-Gaussian parameters
            V_pred, logits_hist_future = model(V_obs, identity)
            V_pred = V_pred.squeeze()

            # [12, N, 2]
            # the GT positions
            V_tr = V_tr.squeeze()


            if True:

                ########################################################
                ## V_pred --- bi-Gaussian prediction
                ## V_tr   --- GT future positions
                ## use bi-Gaussian loss function to optimize
                ########################################################
                l_bivar = graph_loss(V_pred, V_tr)

                #############################
                # interaction loss
                # pairwise relative distance
                # energy loss
                #############################
                # region
                loss_pairwise_rel = torch.tensor(0, device=device).float()
                if use_pairwise_rel > 0 and N_obj > 2:
                    # [12, N] -> [12, N, N]
                    # future gt distance
                    num_valid = num_invalid = 0
                    # hist abs trajectory
                    V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
                    V_x_last = torch.from_numpy(V_x[-1:, :N_obj, :]).float().to(device)
                    # V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                    #                                         V_x[-1, :, :].copy())
                    # V_y_rel_to_abs_tensor = torch.from_numpy(V_y_rel_to_abs).to(device)
                    V_gt_fut_abs_tensor = nodes_rel_to_nodes_abs_tensor(V_tr, V_x_last)
                    # print(V_gt_fut_abs_tensor.size())

                    energy_bin_gt, fut_dist_gt = calc_binary_path_energy(two_sigma_square, V_gt_fut_abs_tensor,
                                                                         to_sum=False)
                    # [12, N, 2]
                    pred_xy = V_pred[:, :, 0:2]
                    V_pred_abs_tensor = nodes_rel_to_nodes_abs_tensor(pred_xy, V_x_last)
                    energy_bin_pred, pred_dist_gt = calc_binary_path_energy(two_sigma_square, V_pred_abs_tensor,
                                                                            to_sum=False)

                    small_dist_index = torch.logical_or(fut_dist_gt <= 3 * args.social_dist_sigma,
                                                        pred_dist_gt <= 3 * args.social_dist_sigma)
                    nonzero_dist_index = torch.logical_and(fut_dist_gt > 0, small_dist_index)
                    subsequent_mask = torch.triu(torch.ones((1, N_obj, N_obj), device=device), diagonal=1).bool()
                    all_dist_index = torch.logical_and(nonzero_dist_index, subsequent_mask)
                    # print(energy_bin.size(), pred_xy_kernel.size(), all_dist_index.size())
                    assert all_dist_index.size(0) == energy_bin_gt.size(0) == pred_seq_len

                    for ni in range(pred_seq_len):
                        msk = all_dist_index[ni, ...]
                        x_tensor = energy_bin_pred[ni, ...][msk]
                        n_valid_pair = x_tensor.size(0)
                        # assert n_valid_pair <= (N_obj - 1) * N_obj // 2
                        if n_valid_pair > 0:
                            energy_t = energy_bin_gt[ni, ...][msk]
                            seq_sorted, seq_true_indices = torch.sort(energy_t)

                            y_tensor = torch.linspace(0, 1, n_valid_pair, device=device)
                            a_tensor = torch.ones(n_valid_pair, device=device) / n_valid_pair
                            b_tensor = torch.ones(n_valid_pair, device=device) / n_valid_pair
                            # Robust rank sort
                            R, S = Rank_Sort_log(a_tensor, b_tensor, x_tensor, y_tensor, eps=args.eps, param1=p1,
                                                 param2=min_std)
                            seq_pred_rank = R - 1
                            # assert not torch.any(torch.isnan(S)), 'NaN in Rank result'
                            if torch.any(torch.isnan(S)):
                                # print('Skip NaN in Rank')
                                num_invalid += 1
                                continue
                            num_valid += 1

                            l_sort_ind = pairwise_loss(seq_pred_rank, seq_true_indices)
                            if l_sort_ind.item() < 10.0:
                                loss_pairwise_rel += l_sort_ind

                    if num_valid > 0:
                        loss_pairwise_rel /= num_valid

                # endregion

                ######################
                # CLIP loss
                ######################
                loss_clip = torch.tensor(0, device=device).float()
                if use_clip:
                    if N_obj >= 2:
                        # calculate intra-loss
                        # the idea is that: the move pattern of the predicted future traj,
                        # should be closer to its history data, compared with others
                        # therefore, we use CLIP to contrastive learning
                        label_clip = torch.arange(N_obj, device=device).detach()
                        loss_hist_pred_clip = F.cross_entropy(logits_hist_future, label_clip)
                        loss_pred_hist_clip = F.cross_entropy(logits_hist_future.T, label_clip)
                        # perform CLIP learning
                        loss_clip = 0.5 * (loss_hist_pred_clip + loss_pred_hist_clip)

                ######################
                # Total loss #########
                ######################
                a1 = 0.3
                a2 = 0.3
                loss_cum = l_bivar + a1 * loss_clip + a2 * loss_pairwise_rel

                if is_fst_loss:
                    loss = loss_cum
                    is_fst_loss = False
                else:
                    loss += loss_cum

            if batch_count % args.batch_size == 0:
                cur_iter += 1
                loss = loss / args.batch_size
                is_fst_loss = True
                loss.backward()

                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                optimizer.step()
                # Metrics
                loss_batch += loss.item()
                # print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)
                print('TRAIN:', '\t Epoch:', epoch, '\t Batch', batch_count, '\t Loss:',
                      np.round(loss_batch / batch_count, 5), '\t lr:', optimizer.param_groups[0]["lr"])
                print('Total loss', np.round(loss_cum.item(), 2),
                      'Bivariate loss', np.round(l_bivar.item(), 2),
                      'Clip loss', np.round(a1 * loss_clip.item(), 2),
                      'Pairwise relative loss', np.round(a2 * loss_pairwise_rel.item(), 2))
                if use_pairwise_rel:
                    print('Num valid/invalid %d/%d' % (num_valid, num_invalid))

        metrics['train_loss'].append(loss_batch / batch_count)


    def vald(epoch, model, checkpoint_dir, loader_val):
        global metrics, constant_metrics
        model.eval()
        loss_batch = 0
        batch_count = 0
        is_fst_loss = True
        loader_len = len(loader_val)
        # turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

        for cnt, batch in enumerate(loader_val):
            batch_count += 1
            # Get data
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, V_tr = batch

            with torch.no_grad():
                N_obj = V_obs.size(2)
                identity = get_ID(obs_seq_len, N_obj, device)
                V_pred = model(V_obs, identity, use_train=False)  # A_obs <8, #, #>

                V_pred = V_pred.squeeze()
                V_tr = V_tr.squeeze()

                loss = bivariate_loss(V_pred, V_tr)
                loss_batch += loss.item()

                if batch_count % 100 == 0:
                    print('VALD:', '\t Epoch:', epoch, '\t Iter %d/%d' % (batch_count, loader_len),
                          '\t Loss:', loss_batch / batch_count)

        metrics['val_loss'].append(loss_batch / batch_count)

        if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
            constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
            constant_metrics['min_val_epoch'] = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'val_best.pth'))
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'val_best_%03d.pth' % (epoch,)))
            print('****** Found best model at epoch %d with min_val_loss %.3f' % (epoch, metrics['val_loss'][-1]))
            print('Save to %s' % (checkpoint_dir,))
        else:
            print('----- Current loss %.3f' % (metrics['val_loss'][-1],))
            print('----- Min     loss %.3f  in epoch %d, ' % (
                constant_metrics['min_val_loss'], constant_metrics['min_val_epoch']))
            print('Save folder %s' % (checkpoint_dir,))

        if epoch % 5 == 0:
            print('Save at %s at epoch %d' % (checkpoint_dir, epoch))
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'val_ep%02d.pth' % (epoch,)))


    # region train
    for epoch in range(args.num_epochs):
        train(epoch, model, optimizer, checkpoint_dir, loader_train)
        vald(epoch, model, checkpoint_dir, loader_val)

        if args.use_lrschd:
            scheduler.step()

        print('*' * 30)
        print('Epoch:', args.dataset + '/' + args.tag, ":", epoch, "/", args.num_epochs)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])

        print(constant_metrics)
        print('*' * 30)

        with open(os.path.join(checkpoint_dir, 'constant_metrics.pkl'), 'wb') as fp:
            pickle.dump(constant_metrics, fp)
    # endregion
