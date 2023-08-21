import argparse
import copy
import json
import os
import time

from tensorboardX import SummaryWriter

import dialog_config
from utils.utils import *
from ReplayBuffer.ReplayBuffer import *
from KGData.KG import KGADJ
from Agent.GAT import GAT
from Agent.KGRL import Agent
from UserSim.UserSimulator import UserSimulator
from dialog_system.dialog_manager import DialogManager
from DiseaseInference.DisInfer import DiseaseInference


current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path))
model_folder = os.path.join(father_path, 'output', 'kgrl_model', '2023.08.11-23-38-51', 'kgrl_model_0.6997.pth.tar')

timeStr = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
writer = SummaryWriter('./runs/{}'.format('KGRL-' + timeStr))
parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', dest='data_folder', type=str, default='./KGData/dataset_cmd/', help='folder for all data')
parser.add_argument('--max_turn', dest='max_turn', default=40, type=int, help='maximum length of each dialog')
parser.add_argument('--episodes', dest='episodes', default=1000, type=int, help='Total number of episodes to run')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=0., help='Epsilon to determine stochasticity(随机性) of epsilon-greedy agent policies')

parser.add_argument('--buffer_size', dest='buffer_size', type=int, default=30000, help='the size for experience replay')
parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=128, help='the hidden size for DQN')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='batch size')
parser.add_argument('--input_shape', dest='input_shape', type=int, default=100, help='input_shape')  # 先用100
parser.add_argument('--sym_shape', dest='sym_shape', type=int, default=358, help='sym_shape')  # num for symptom
parser.add_argument('--attention_dim', dest='attention_dim', type=int, default=128, help='attention_dim')  # attention dim
parser.add_argument('--dis_shape', dest='dis_shape', type=int, default=27, help='dis_shape')  # num for disease 是否加UNK？？
parser.add_argument('--entity_num', dest='entity_num', type=int, default=385, help='entity_num')  # num for kg entity
parser.add_argument('--embed_size', dest='embed_size', type=int, default=100, help='embed_size')  # size for embedding
parser.add_argument('--dropout', dest='dropout', type=float, default=0., help='dropout')  # dropout for gat
parser.add_argument('--alpha', dest='alpha', type=float, default=0.2, help='alpha')  # alpha for gat
parser.add_argument('--heads', dest='heads', type=int, default=2, help='heads')  # heads for gat
parser.add_argument('--threshold', dest='threshold', type=float, default=0.9, help='threshold')  # threshold for disease inference
parser.add_argument('--set_turns', dest='set_turns', type=int, default=2, help='set_turns')  # set_truns
parser.add_argument('--disease_model', dest='disease_model', type=str, default='./output/useful_models/disease_inference_0.7079.pth.tar', help='trained disease model')  # pretrained model for disease inference
parser.add_argument('--gat_model', dest='gat_model', type=str, default='./output/useful_models/gat_0.6960.pth.tar', help='trained gat model')  # pretrained model for gat
parser.add_argument('--save_model_path', dest='save_model_path', type=str, default='./output/kgrl_model/' + str(timeStr) + '/', help='path for saving model')  # path for saving model
parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='lr for model')
parser.add_argument('--discount', dest='discount', type=float, default=0.9, help='discount for agent')
parser.add_argument('--target_net_update_freq', dest='target_net_update_freq', type=int, default=10, help='update frequency')

parser.add_argument('--simulation_epoch_size', dest='simulation_epoch_size', type=int, default=100, help='the size of validation set')
parser.add_argument('--warm_start', dest='warm_start', type=int, default=1, help='0: no warm start; 1: warm start for training')
parser.add_argument('--warm_start_epochs', dest='warm_start_epochs', type=int, default=5000, help='the number of epochs for warm start')
parser.add_argument('--learning_phase', dest='learning_phase', default='train', type=str, help='train/test; default is all')
parser.add_argument('--train_set', dest='train_set', default='train', type=str, help='train/test/all; default is all')
parser.add_argument('--test_set', dest='test_set', default='test', type=str, help='train/test/all; default is all')

args = parser.parse_args()
params = vars(args)

print('Parameters:')
print(json.dumps(params, indent=2))

data_folder = params['data_folder']
goal_set = load_pickle('{}/goal_cmd.pk'.format(data_folder))
slot_set = text_to_dict('{}/cmd_slot_set.txt'.format(data_folder))  # all slots with symptoms + all disease
slot_list = text_to_list('{}/cmd_slot_set.txt'.format(data_folder))
sym_dict = text_to_dict('{}/symptoms_cmd.txt'.format(data_folder))  # all symptoms
dise_dict = text_to_dict('{}/diseases_cmd.txt'.format(data_folder))  # all diseases
req_dise_sym_dict = load_pickle('{}/req_dise_sym_dict.p'.format(data_folder))
# dise_sym_num_dict = load_pickle('{}/dise_sym_num_high_dict(≥3).p'.format(data_folder))
dise_sym_num_dict = load_pickle('{}/dise_sym_num_dict.p'.format(data_folder))

learning_phase = params['learning_phase']  # train
train_set = params['train_set']  # train
test_set = params['test_set']  # test
max_turn = params['max_turn']  # 40
num_episodes = params['episodes']  # 1000
warm_episodes = params['warm_start_epochs']
simulation_episodes = params['simulation_epoch_size']

################################################################################
#   Parameters for Agents
################################################################################
disease_model = DiseaseInference(params['attention_dim'], params['dis_shape'], params['embed_size'], params['sym_shape'], dialog_config.device)
Buffer = ReplayBuffer(params['buffer_size'], dialog_config.device)
gat = GAT(params['embed_size'], params['embed_size'], params['embed_size'], params['dropout'], params['alpha'], params['heads'], dialog_config.device, params['entity_num'], params['embed_size'], params['dis_shape'])
model = Agent(gat, disease_model, params['input_shape'], params['sym_shape'], params['dis_shape'], params['hidden_size'], slot_set, slot_list, params['entity_num'],
              params['embed_size'], params['threshold'], params['epsilon'], Buffer, req_dise_sym_dict, dise_sym_num_dict, params['disease_model'], params['gat_model'], params['lr'],
              params['batch_size'], params['target_net_update_freq'], params['warm_start'], params['buffer_size'], params['discount'], max_turn, params['set_turns'])

################################################################################
#   Parameters for User Simulators
################################################################################
usersim_params = {}
usersim_params['max_turn'] = max_turn
usersim_params['use_mode'] = params['learning_phase']

user_sim = UserSimulator(sym_dict, goal_set, usersim_params)

################################################################################
# Dialog Manager
################################################################################
kg = KGADJ(dialog_config.device)
dialog_manager = DialogManager(model, user_sim, kg)

simulation_epoch_size = params['simulation_epoch_size']

best_res = {'success_rate': 0, 'ave_reward': float('-inf'), 'ave_turns': float('inf')}


def load_model(path, all_model):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    all_model.load_state_dict(checkpoint['state_dict'])


def test(simu_size, data_type):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    user_sim.use_mode = data_type
    user_sim.test = 1
    user_sim.test_goal = copy.deepcopy(user_sim.goals_set[data_type])
    res = {}
    avg_hit_rate = 0.0
    model.epsilon = 0.
    epoch = 0
    for episode in range(simu_size):
        dialog_manager.initialize_episode()
        episode_over = False
        episode_hit_rate = 0
        while not episode_over:
            episode_over, r, dialog_status, hit_rate = dialog_manager.next_dialog()
            cumulative_reward += r
            if episode_over:
                # if reward > 0:
                episode_hit_rate += hit_rate
                if dialog_status == dialog_config.SUCCESS_DIALOG:
                    successes += 1
                    print("%s simulation episode %s: Success" % (data_type, episode))
                    with open(dialog_config.predict_log_re, 'a+', encoding='utf-8') as ref:
                        ref.write("%s simulation episode %s: Success \n" % (data_type, episode))

                else:
                    # pass
                    print("%s simulation episode %s: Fail" % (data_type, episode))
                    with open(dialog_config.predict_log_re, 'a+', encoding='utf-8') as ref:
                        ref.write("%s simulation episode %s: Fail \n" % (data_type, episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count
                if dialog_manager.state_tracker.turn_count > 3:
                    episode_hit_rate /= (dialog_manager.state_tracker.turn_count - 3) * 0.5
                    avg_hit_rate += episode_hit_rate
                    epoch += 1
                    print(episode_hit_rate)
                    with open(dialog_config.predict_log_re, 'a+', encoding='utf-8') as ref:
                        ref.write("episode_hit_rate: %s \n" % (episode_hit_rate))
    res['success_rate'] = float(successes) / float(simu_size)
    res['ave_reward'] = float(cumulative_reward) / float(simu_size)
    res['ave_turns'] = float(cumulative_turns) / float(simu_size)
    avg_hit_rate = avg_hit_rate / epoch
    print("%s hit rate %.4f, success rate %.4f, ave reward %.4f, ave turns %.4f" % (
    data_type, avg_hit_rate, res['success_rate'], res['ave_reward'], res['ave_turns']))
    with open(dialog_config.predict_log_re, 'a+', encoding='utf-8') as ref:
        ref.write("%s hit rate %.4f, success rate %.4f, ave reward %.4f, ave turns %.4f \n" % (
    data_type, avg_hit_rate, res['success_rate'], res['ave_reward'], res['ave_turns']))
    model.epsilon = params['epsilon']
    return res


# load_model(model_folder, model)
model.warm_start = 2  # 开始用模型
model.mode = 'test'
with open(dialog_config.predict_log_re, 'a+', encoding='utf-8') as ref:
    ref.write(str(model_folder) + '###############\n')
    ref.write('#############'+ 'Threhold:' + str(params['threshold']) + 'Turn:' + str(params['set_turns']) +'###############\n')
eval_res = test(len(goal_set['test']), test_set)