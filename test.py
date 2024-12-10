import pickle
import glob
import argparse
import glob
from torch.utils.data.dataloader import DataLoader
import torch.distributions.multivariate_normal as torchdist

from tools.data_utils import *
from tools.metrics import *
from model import TrajectoryModel
import copy
from loader import TrajectoryDataset
from tools.utils import get_ID


def test(KSTEPS=20, use_visualize=False):
    model.eval()
    raw_data_dict = {}
    ade_bigls = []
    fde_bigls = []
    int_pol_rt_bigls = []
    args.social_dist_sigma = 1
    two_sigma_square = 2 * args.social_dist_sigma ** 2

    step = 0
    for batch in loader_test:

        if step % 100 == 0:
            print('%d/%d' % (step, len(loader_test)))

        step += 1
        # Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch

        N_obj = V_obs.size(2)
        identity = get_ID(obs_seq_len, N_obj, device)

        V_pred = model(V_obs, identity, use_train=False)  # A_obs <8, #, #>

        V_pred = V_pred.squeeze()
        V_tr = V_tr.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
        #
        # #For now I have my bi-variate parameters
        # #normx =  V_pred[:,:,0:1]
        # #normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr
        #
        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).to(device)
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)
        #
        #
        # ### Rel to abs
        # ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len
        #
        # #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        int_pol_rt_ls = []
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs[:, :, :, 1:3].data.cpu().numpy().squeeze().copy(),
                                                V_x[0, :, :].copy())
        #
        # V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                V_x[-1, :, :].copy())
        V_y_rel_to_abs_tensor = torch.from_numpy(V_y_rel_to_abs).to(device)

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        #
        for k in range(KSTEPS):

            V_pred = mvnormal.sample()

            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                       V_x[-1, :, :].copy())

            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

            ######################
            # intimacy/politeness
            ######################
            if use_intimacy:
                # [12, N, 2=xy]
                V_pred_rel_to_abs_tensor = torch.from_numpy(V_pred_rel_to_abs).to(device)
                V_pred_abs_diff = V_pred_rel_to_abs_tensor.unsqueeze(2) - V_pred_rel_to_abs_tensor.unsqueeze(1)
                V_gt_abs_diff = V_y_rel_to_abs_tensor.unsqueeze(2) - V_y_rel_to_abs_tensor.unsqueeze(1)
                intimacy_score, politeness_score = intimacy_politeness_score(V_pred_abs_diff, V_gt_abs_diff,
                                                                             args.social_dist_sigma, hinge_ratio=0.25)
                # print(intimacy_score, politeness_score)
                # int_pol_rt_ls.append(0.5 * (intimacy_score + politeness_score))
                score_n = 0
                score_sum = 0
                if intimacy_score > 0:
                    score_sum += intimacy_score
                    score_n += 1
                if politeness_score > 0:
                    score_sum += politeness_score
                    score_n += 1
                if score_n > 0:
                    score_sum /= score_n
                int_pol_rt_ls.append(score_sum)

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

        if use_intimacy:
            int_pol_rt_bigls.append(max(int_pol_rt_ls))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)

    if use_intimacy:
        int_pol_rt_ = sum(int_pol_rt_bigls) / len(int_pol_rt_bigls)
    else:
        int_pol_rt_ = -100.0

    return ade_, fde_, raw_data_dict, int_pol_rt_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('-d', '--dataset', default='',
                        help='eth,hotel,univ,zara1,zara2')
    parser.add_argument('-p', '--model_dir', type=str, default='',
                        help='path of test models')
    parser.add_argument('--social_dist_sigma', type=int, default=1,
                        help='social distance, set to 2m')
    parser.add_argument('-ui', '--use_intimacy', action="store_true", default=True,
                        help='evaluate with intimacy score')

    args = parser.parse_args()

    KSTEPS = 20

    # checkpoint_path = './checkpoint/%s'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_intimacy = args.use_intimacy

    torch.manual_seed(1357)
    torch.cuda.manual_seed(1357)
    np.random.seed(1357)

    exp_path = args.model_dir
    assert len(exp_path) > 0
    print("*" * 50)
    print("Evaluating model:", exp_path)

    if exp_path.endswith('/'):
        exp_path = exp_path[:-1]
    t1 = exp_path.split('/')[-1]
    t2 = t1.split('_')[0]

    dset = t2
    if len(args.dataset) > 0:
        assert args.dataset == dset
    print('Model being tested are:', dset)

    if exp_path.endswith('.pth'):
        exp_dir = os.path.dirname(exp_path)
        args_path = os.path.join(exp_dir, 'args.pkl')
    else:
        args_path = os.path.join(exp_path, 'args.pkl')

    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    # Data prep
    obs_seq_len = args.obs_len
    pred_seq_len = args.pred_len

    data_set = './datasets/' + args.dataset + '/'
    processed_data_set = './dataset_processed/' + args.dataset + '/'
    os.makedirs(processed_data_set, exist_ok=True)

    dset_test = TrajectoryDataset(
        data_set + 'test/',
        processed_data_set + 'test.pkl',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_test = DataLoader(
        dset_test,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=1)

    model = TrajectoryModel(number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                            obs_len=8, pred_len=12, n_tcn=5, out_dims=5).to(device)

    # if it's a model file
    if exp_path.endswith('.pth'):
        assert os.path.isfile(exp_path)
        model.load_state_dict(torch.load(exp_path))
    else:
        model_path = os.path.join(exp_path, 'val_best.pth')
        if os.path.isfile(model_path):
            if use_cuda:
                model.load_state_dict(torch.load(model_path), strict=False)
            else:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
            print('Load %s' % (model_path,))
        else:
            print('Warning!!!! No model found')

    ade_ = 1e8
    fde_ = 1e8
    int_pol_rt_ = -1e8
    print("Testing ....")
    ad, fd, raw_data_dic_, int_pol_rt = test()
    ade_ = min(ade_, ad)
    fde_ = min(fde_, fd)
    int_pol_rt_ = max(int_pol_rt_, int_pol_rt)
    # print("ADE:", np.round(ade_, 2), " FDE:", np.round(fde_, 2))
    print(np.round(ade_, 3), " ", np.round(fde_, 3), " ", np.round(int_pol_rt_, 3))

    print("*" * 50)
