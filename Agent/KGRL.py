import copy
import os.path

import networkx as nx
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import dialog_config
import m_d


class KGRL(nn.Module):

    def __init__(self, GraphAgent, DisInfer, input_shape, sym_shape, dis_shape, hidden_size, slot_set, entity_num, embed_size, threshold, device, max_turn, set_turns):
        super(KGRL, self).__init__()
        self.kg_model = GraphAgent
        self.slot_set = slot_set
        self.entity_num = entity_num
        self.embed_size = embed_size

        self.input_shape = input_shape
        self.sym_shape = sym_shape
        self.dis_shape = dis_shape  # 是否加"UNK"??
        self.hidden_size = hidden_size

        self.threshold = threshold
        self.device = device
        self.max_turn = max_turn

        self.set_turns = set_turns

        # self.Dis_Inference = nn.Sequential(  # 疾病推理模块
        #     nn.Linear(self.input_shape, self.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size, self.dis_shape),
        #     nn.Softmax()
        # )  # 离线训练
        self.Dis_Inference = DisInfer

        self.Sym_Inquiry = nn.Sequential(  # 症状检查模块
            nn.Linear(self.input_shape, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.sym_shape),
            # nn.Softmax()
        )

        self.mu_ = nn.Sequential(
            nn.Linear(self.dis_shape + self.sym_shape, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

        self.m_d = copy.deepcopy(m_d.tfidf_matrix_).to(self.device)

        self.current_slots = {}

    def filter_sig(self, tfidf_p):
        for b in range(len(tfidf_p)):
            for i in range(len(tfidf_p[b])):
                if tfidf_p[b][i] > 0.:
                    tfidf_p[b][i] = F.sigmoid(tfidf_p[b][i])
        return tfidf_p

    def forward(self, dialog_state, dialog_sym_rep, kg_adj, sym_flag, sym_mask, disease_mask, symptoms_mask):
        dialog_state_ = self.get_sym_mask(dialog_state)
        with torch.no_grad():
            graph_rep = self.kg_model.predict(kg_adj)  # N*100
            state = torch.matmul(dialog_sym_rep, graph_rep).squeeze(1)  # (b, 1, N) * (b, N, 100) → (b, 1, 100)
            disease_ = self.Dis_Inference.predict(sym_mask)
            disease_dis = disease_ * disease_mask
        mu = F.sigmoid(self.mu_(dialog_state_))
        sym_tfidf_prob = torch.matmul(disease_dis, self.m_d)  # tf-idf
        sym_tf_prob = self.filter_sig(sym_tfidf_prob)
        symptom_p = F.sigmoid(self.Sym_Inquiry(state))  # RL
        sym_prob = mu * symptom_p + (torch.ones_like(mu) - mu) * sym_tf_prob
        symptom_ = sym_prob * sym_flag
        symptom_final = symptom_ * symptoms_mask

        return disease_dis, symptom_final

    def predict(self, kg_matrix, state, dialog_sym_rep, sym_flag, sym_mask, turn, disease_mask, symptoms_mask):
        # with torch.no_grad():
        dis, sym = self.forward(state, dialog_sym_rep, kg_matrix, sym_flag, sym_mask, disease_mask, symptoms_mask)
        dis_pro = dis.max(1)[0].view(1, 1).item()
        dis_predict = dis.max(1)[1].view(1, 1).item()
        sym_index = sym.max(1)[1].view(1, 1).item()
        sym_pro = sym.max(1)[0].view(-1, 1).item()
        avaliable_dis = torch.sum(disease_mask)
        # if (dis_pro > self.threshold and turn > 5) or turn == self.max_turn or avaliable_dis == 1:  # 训练
        if (dis_pro > self.threshold and turn > self.set_turns) or turn == self.max_turn or avaliable_dis == 1 or dialog_config.include_sym_is_empty:  # 疾病置信度高；症状的概率都不足以询问；到达最大会话次数；均直接诊断
        # if dis_pro > self.threshold or sym_pro < 0.5 or turn == self.max_turn or avaliable_dis == 1:  # 疾病置信度高；症状的概率都不足以询问；到达最大会话次数；均直接诊断
            return dis_predict, 1, dis
        else:
            return sym_index + self.dis_shape, 0, dis

    def get_sym_mask(self, batch_state):  # 用于计算症状是否存在，输入给疾病推理模块
        ones = torch.ones(batch_state.size()).to(self.device)
        zeros = torch.zeros(batch_state.size()).to(self.device)
        state_o = torch.where(batch_state == 1, ones, zeros)  #
        state_ = torch.where(batch_state == -1, -ones, state_o)  #

        return state_



class Agent(nn.Module):

    def __init__(self, GraphAgent, DisInfer, input_shape, sym_shape, dis_shape, hidden_size, slot_set, slot_list, entity_num, embed_size,
                 threshold, epsilon, replaybuffer, req_dise_sym_dict, dise_sym_num_dict, dis_path, gat_path, lr, batch_size, target_net_update_freq, warm_start, buffer_size, discount, max_turn, set_turns):
        super(Agent, self).__init__()
        # self.kg_representation = torch.zeros(46, 3)  # 初始向量
        # self.kg_model = GraphAgent
        self.buffer = replaybuffer

        self.slot_set = slot_set
        self.entity_num = entity_num
        self.embed_size = embed_size
        self.epsilon = epsilon
        self.request_set = copy.deepcopy(dialog_config.sys_request_slots_highfreq)
        self.req_dise_sym_dict = req_dise_sym_dict  # high freq dise sym relations
        self.dise_sym_num_dict = dise_sym_num_dict  # dise sym discrete
        self.slot_list = slot_list

        self.input_shape = input_shape
        self.sym_shape = sym_shape
        self.dis_shape = dis_shape  # 是否加"UNK"??
        self.hidden_size = hidden_size
        self.lr = lr
        self.device = dialog_config.device
        self.batch_size = batch_size
        self.target_net_update_freq = target_net_update_freq
        self.warm_start = warm_start
        self.experience_replay_size = buffer_size
        self.rest_sym = copy.deepcopy(self.slot_list[self.dis_shape:])

        self.discount = discount
        self.update_count = 0
        self.max_turn = max_turn
        self.mode = 'train'

        self.threshold = threshold
        self.turn = 0
        self.current_slots = {}
        self.disc = False

        self.set_turns = set_turns

        self.model = KGRL(GraphAgent, DisInfer, input_shape, sym_shape, dis_shape, hidden_size, slot_set, entity_num, embed_size, threshold, self.device, self.max_turn, self.set_turns)
        self.target_model = KGRL(GraphAgent, DisInfer, input_shape, sym_shape, dis_shape, hidden_size, slot_set, entity_num, embed_size, threshold, self.device, self.max_turn, self.set_turns)
        disinfer_ = torch.load(dis_path, map_location=self.device)
        self.model.Dis_Inference.load_state_dict(disinfer_['state_dict'])  # 加载离线训练模型
        self.target_model.Dis_Inference.load_state_dict(disinfer_['state_dict'])
        gat_ = torch.load(gat_path, map_location=self.device)
        self.model.kg_model.load_state_dict(gat_['state_dict'])  # 加载离线训练模型
        self.target_model.kg_model.load_state_dict(gat_['state_dict'])
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.optimmizer = torch.optim.Adam(self.model.Sym_Inquiry.parameters(), self.lr)

    def initialize_episode(self):
        self.turn = 0
        self.current_slots = {}
        self.rest_sym = copy.deepcopy(self.slot_list[self.dis_shape:])
        for name, param in self.model.Dis_Inference.named_parameters():
            param.requires_grad = False
        for name, param in self.model.kg_model.named_parameters():
            param.requires_grad = False

    def run_policy(self, kg_matrix, state, dialog_context, disease_mask, symptoms_mask):  # 使用DQN
        if np.random.random() < self.epsilon:  # epsilon DQN  随机采样动作
            action_index = np.random.randint(0, self.entity_num)
            sym_flag, dialog_sym_rep = self.get_sym_flag(state[:, self.dis_shape:self.dis_shape + self.sym_shape], state)  # 得到症状是否询问
            sym_mask = self.get_sym_mask(state[:, self.dis_shape:self.dis_shape + self.sym_shape])
            _, _, dis_ = self.model.predict(kg_matrix, state, dialog_sym_rep, sym_flag, sym_mask, self.turn, disease_mask, symptoms_mask)

            ############# record processing information #################
            if self.warm_start == 2 and self.disc == False:
                if self.mode == 'train':
                    with open(dialog_config.diagnosis_re, 'a+', encoding='utf-8') as ref:
                        ref.write('DisInfer:' + str(dis_) + '\n')
                        ref.write('Random')
                elif self.mode == 'test':
                    with open(dialog_config.test_re, 'a+', encoding='utf-8') as ref:
                        ref.write('DisInfer:' + str(dis_) + '\n')
                        ref.write('Random')

            if action_index < self.dis_shape:
                act_slot_response = {'diaact': 'inform', 'inform_slots': {'disease': self.slot_list[action_index]}, 'request_slots': {}}
                return {'act_slot_response': act_slot_response}, dis_
            else:
                act_slot_response = {'diaact': 'request', 'inform_slots': {}, 'request_slots': {self.slot_list[action_index]: 'UNK'}}
                return {'act_slot_response': act_slot_response}, dis_
        else:
            if self.warm_start == 1:  # 在开始时利用规则的方法采取动作
                sym_flag, dialog_sym_rep = self.get_sym_flag(state[:, self.dis_shape:self.dis_shape + self.sym_shape], state)  # 得到症状是否询问
                sym_mask = self.get_sym_mask(state[:, self.dis_shape:self.dis_shape + self.sym_shape])
                _, _, dis_ = self.model.predict(kg_matrix, state, dialog_sym_rep, sym_flag, sym_mask, self.turn, disease_mask, symptoms_mask)
                act_slot_response = self.rule_policy(dialog_context)
                return {'act_slot_response': act_slot_response}, dis_
            else:  # v2修改
                sym_flag, dialog_sym_rep = self.get_sym_flag(state[:, self.dis_shape:self.dis_shape+self.sym_shape], state)  # 得到症状是否询问
                sym_mask = self.get_sym_mask(state[:, self.dis_shape:self.dis_shape+self.sym_shape])
                action_, flag, dis_ = self.model.predict(kg_matrix, state, dialog_sym_rep, sym_flag, sym_mask, self.turn, disease_mask, symptoms_mask)

                ############# record processing information #################
                if self.warm_start == 2 and self.disc == False:
                    if self.mode == 'train':
                        with open(dialog_config.diagnosis_re, 'a+', encoding='utf-8') as ref:
                            ref.write('DisInfer:' + str(dis_) + '\n')
                    elif self.mode == 'test':
                        with open(dialog_config.test_re, 'a+', encoding='utf-8') as ref:
                            ref.write('DisInfer:' + str(dis_) + '\n')

                if flag == 1:  # 做出诊断
                    act_slot_response = {'diaact': 'inform', 'inform_slots': {'disease': self.slot_list[action_]}, 'request_slots':{}}
                    return {'act_slot_response': act_slot_response}, dis_
                elif flag == 0:  # 查询症状, 不在此做重复判断
                    act_slot_response = {'diaact': 'request', 'inform_slots': {}, 'request_slots': {self.slot_list[action_]: 'UNK'}}
                    return {'act_slot_response': act_slot_response}, dis_

    def state_to_action(self, kg_matrix, dialog_context, disease_mask, symptoms_mask):
        if not self.disc:
            self.turn += 2
        self.state_slot = self.state_representation(dialog_context)  # 得到疾病症状对话向量 (0, 0, 0, 0, 0, 0.3, 0.3, 0.6, 1, ....)
        self.action, disease_distribution = self.run_policy(kg_matrix, self.state_slot, dialog_context, disease_mask, symptoms_mask)
        return self.action, disease_distribution

    def state_representation(self, dialog_context):
        state_slot = torch.zeros(1, self.entity_num).to(self.device)  # 实体大小由疾病和症状组成
        current_slots = dialog_context['current_slots']
        current_slots_ = []
        for slot in current_slots['inform_slots']:
            if slot != 'disease' and slot not in self.slot_set:
                continue

            if slot == 'disease':
                state_slot[0, self.slot_set[current_slots['inform_slots']['disease']]] = 1
            else:
                state_slot[0, self.slot_set[slot]] = current_slots['inform_slots'][slot]
                current_slots_.append(slot)

        # 还没提问的症状
        for sym in self.rest_sym:
            if sym not in current_slots_:
                state_slot[0, self.slot_set[sym]] = dialog_config.NOT_MENTION

        return state_slot

    def get_sym_flag(self, batch_state, total_state):  # 得到那些症状问过，哪些没问过，以便后续过滤  疾病设为1× 无需疾病
        ones = torch.ones(batch_state.size()).to(self.device)
        zeros = torch.zeros(batch_state.size()).to(self.device)
        ones_ = torch.ones(total_state.size()).to(self.device)
        zeros_ = torch.zeros(total_state.size()).to(self.device)
        return torch.where(batch_state == 0.3, ones, zeros), torch.where(total_state == 1, ones_, zeros_).unsqueeze(1)

    def get_sym_mask(self, batch_state):  # 用于计算症状是否存在，输入给疾病推理模块

        ones = torch.ones(batch_state.size()).to(self.device)
        zeros = torch.zeros(batch_state.size()).to(self.device)
        state_ = torch.where(batch_state == 1, ones, zeros)
        # state_o = torch.where(batch_state == 1, ones, zeros)  #
        # state_ = torch.where(batch_state == -1, -ones, state_o)  #
        size = len(batch_state.size())
        mask_ = torch.unsqueeze(state_, size)
        mask_ = mask_.repeat(1, 1, self.embed_size).to(self.device)

        return mask_

    def train(self, train=True):
        cur_bellman_err = 0.0
        for iter in range(int(self.buffer.size/self.batch_size)):
            # sample experience datas
            transitions, indices, weights = self.buffer.sample(self.batch_size)
            # print('transition:', transitions)
            dialog_state, kg_matrix, disease_mask, symptoms_mask, agent_action, next_dialog_state, next_kg_matrix, next_disease_mask, next_symptoms_mask, reward, episode_over = zip(*transitions)
            batch_state = torch.cat(dialog_state, dim=0).view(-1, self.entity_num).to(self.device)
            batch_kg_matrix = torch.cat(kg_matrix, dim=0).view(-1, self.entity_num, self.entity_num).to(self.device)
            batch_action = torch.tensor(agent_action, device=self.device).squeeze().view(-1, 1)
            batch_disease_mask = torch.cat(disease_mask, dim=0).view(-1, self.dis_shape).to(self.device)
            batch_symptoms_mask = torch.cat(symptoms_mask, dim=0).view(-1, self.sym_shape).to(self.device)
            batch_reward = torch.tensor(reward, device=self.device).squeeze().view(-1, 1)
            batch_next_state = torch.cat(next_dialog_state, dim=0).view(-1, self.entity_num).to(self.device)
            batch_next_kg_matrix = torch.cat(next_kg_matrix, dim=0).view(-1, self.entity_num, self.entity_num).to(self.device)
            batch_next_disease_mask = torch.cat(next_disease_mask, dim=0).view(-1, self.dis_shape).to(self.device)
            batch_next_symptoms_mask = torch.cat(next_symptoms_mask, dim=0).view(-1, self.sym_shape).to(self.device)
            batch_episode_over = torch.tensor(episode_over, device=self.device).squeeze().view(-1, 1)
            loss = self.compute_loss(batch_state, batch_kg_matrix, batch_disease_mask, batch_symptoms_mask, batch_action, batch_next_state, batch_next_kg_matrix, batch_next_disease_mask, batch_next_symptoms_mask, batch_reward, batch_episode_over)

            self.optimmizer.zero_grad()
            loss.backward()
            self.optimmizer.step()

            cur_bellman_err += loss.item()
        # print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) * self.batch_size / self.buffer.size, self.buffer.size))
        print("cur bellman err %.4f" % (float(cur_bellman_err)))

        current_path = os.path.abspath(__file__)
        father_path = os.path.abspath(os.path.dirname(current_path))
        father_path_ = os.path.abspath(os.path.dirname(father_path))
        loss_path = os.path.join(father_path_, 'output', 'loss_file.txt')
        with open(loss_path, 'a+', encoding='utf-8') as lf:
            # lf.write("cur bellman err %.4f, experience replay pool %s\n" % (float(cur_bellman_err) * self.batch_size / self.buffer.size, self.buffer.size))
            lf.write("cur bellman err %.4f\n" % (float(cur_bellman_err)))

        self.update_target_model()

    def compute_loss(self, state, kg_matrix, disease_mask, symptoms_mask, action, next_state, next_kg_matrix, next_disease_mask, next_symptoms_mask, reward, episode_over):
        dialog_state, kg_matrix, disease_mask, symptoms_mask, agent_action, next_dialog_state, next_kg_matrix, next_disease_mask, next_symptoms_mask, reward, episode_over = state, kg_matrix, disease_mask, symptoms_mask, action, next_state, next_kg_matrix, next_disease_mask, next_symptoms_mask, reward, episode_over
        sym_flag, dialog_sym_rep = self.get_sym_flag(dialog_state[:, self.dis_shape:self.dis_shape+self.sym_shape], dialog_state)
        sym_mask = self.get_sym_mask(dialog_state[:, self.dis_shape:self.dis_shape + self.sym_shape])
        disease_value, sym_value = self.model(dialog_state, dialog_sym_rep, kg_matrix, sym_flag, sym_mask, disease_mask, symptoms_mask)
        action_value = torch.cat((disease_value, sym_value), dim=1)
        action_value_ = action_value.gather(1, agent_action)
        with torch.no_grad():
            next_sym_flag, next_dialog_sym_rep = self.get_sym_flag(next_dialog_state[:, self.dis_shape:self.dis_shape+self.sym_shape], next_dialog_state)
            next_sym_mask = self.get_sym_mask(next_dialog_state[:, self.dis_shape:self.dis_shape + self.sym_shape])
            next_disease_value, next_sym_value = self.target_model(next_dialog_state, next_dialog_sym_rep, next_kg_matrix, next_sym_flag, next_sym_mask, next_disease_mask, next_symptoms_mask)
            max_next_sym_value = next_sym_value.max(dim=1)[0].view(-1, 1)
        y_ = reward + max_next_sym_value * self.discount * (1. - episode_over)
        # 这里使用哪个值  训练症状检查，因此做它的loss
        loss = torch.mean(F.mse_loss(y_, action_value_))

        return loss

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            print("update target model!!!")
            self.target_model.load_state_dict(self.model.state_dict())

    def register_experience_replay_tuple(self, state, kg_matrix, disease_mask, symptoms_mask, action, next_state, next_kg_matrix, next_disease_mask, next_symptoms_mask, reward, episode_over):
        state_t_rep = self.state_representation(state)
        action_t = 0
        if action['diaact'] == 'request':
            # action_t = self.slot_set.index(list(action['request_slots'].keys())[0])  # 定位动作索引
            action_t = self.slot_set[list(action['request_slots'].keys())[0]]  # 定位动作索引
        elif action['diaact'] == 'inform':
            # action_t = self.slot_set.index(action['inform_slots']['disease'])
            action_t = self.slot_set[action['inform_slots']['disease']]
        next_state_rep = self.state_representation(next_state)
        reward_t = reward
        if episode_over:
            episode_over = 1.
        else:
            episode_over = 0.
        tuple_ = (state_t_rep, kg_matrix, disease_mask, symptoms_mask, action_t, next_state_rep, next_kg_matrix, next_disease_mask, next_symptoms_mask, reward_t, episode_over)

        self.buffer.add(tuple_)
        if self.warm_start == 2:
            with open(dialog_config.buffer_re, 'a+', encoding='utf-8') as ref:
                ref.write('state:' + str(state['current_slots']['inform_slots']) + '\n')
                ref.write('state_t_rep:' + str(state_t_rep) + '\n')
                ref.write('disease_mask:' + str(disease_mask) + '\n')
                ref.write('symptoms_mask:' + str(symptoms_mask) + '\n')
                ref.write('action_t:' + str(action_t) + '\n')
                ref.write('next_state:' + str(next_state['current_slots']['inform_slots']) + '\n')
                ref.write('next_state_rep:' + str(next_state_rep) + '\n')
                ref.write('next_disease_mask:' + str(next_disease_mask) + '\n')
                ref.write('next_symptoms_mask:' + str(next_symptoms_mask) + '\n')
                ref.write('reward_t:' + str(reward_t) + '\n')
                ref.write('episode_over:' + str(episode_over) + '\n')
                ref.write('\n')

    def rule_policy(self, dialog_context):
        """ Rule Policy """
        current_slots = dialog_context['current_slots']
        act_slot_response = {}
        sym_flag = 1  # 1 for no left sym, 0 for still have
        for sym in self.request_set:
            if sym not in current_slots['inform_slots'].keys():
                sym_flag = 0
        dise = self.disease_from_dict(current_slots, sym_flag)
        if dise == dialog_config.NO_MATCH:  # no match but still have syms to ask
            cur_dise_sym_rate = {}
            for dise in self.dise_sym_num_dict:
                if dise not in cur_dise_sym_rate:
                    cur_dise_sym_rate[dise] = 0
                tmp = [v for v in self.dise_sym_num_dict[dise].keys() if v in current_slots['inform_slots'].keys()]
                tmp_sum = 0
                dise_sym_sum = 0
                for sym in tmp:
                    tmp_sum += self.dise_sym_num_dict[dise][sym]
                for sym in self.dise_sym_num_dict[dise]:
                    dise_sym_sum += self.dise_sym_num_dict[dise][sym]
                    # dise_sym_rate[dise] = float(len(tmp))/float(len(self.dise_sym_num_dict[dise]))
                    cur_dise_sym_rate[dise] = float(tmp_sum) / float(dise_sym_sum)

            sorted_dise = list(dict(sorted(cur_dise_sym_rate.items(), key=lambda d: d[1], reverse=True)).keys())
            left_set = []
            for i in range(len(sorted_dise)):
                max_dise = sorted_dise[i]
                left_set = [v for v in self.req_dise_sym_dict[max_dise] if
                            v not in current_slots['inform_slots'].keys()]
                if len(left_set) > 0: break
            # if syms in request set of all disease have been asked, choose one sym in request set
            if len(left_set) == 0:
                print('this will not happen')
                left_set = [v for v in self.request_set if v not in current_slots['inform_slots'].keys()]
            slot = np.random.choice(left_set)
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}

        elif dise == dialog_config.NO_MATCH_BY_RATE:  # no match and no sym to ask
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {'disease': 'UNK'}
            act_slot_response['request_slots'] = {},

        else:  # match one dise by complete match or by rate
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {'disease': dise}
            act_slot_response['request_slots'] = {}

        return act_slot_response

    def disease_from_dict(self, current_slots, sym_flag):

        if sym_flag == 0:
            dise = dialog_config.NO_MATCH
            for d in self.req_dise_sym_dict:
                dise = d
                for sym in self.req_dise_sym_dict[d]:
                    if sym not in current_slots['inform_slots'] or current_slots['inform_slots'][sym] != True:
                        dise = dialog_config.NO_MATCH
                if dise != dialog_config.NO_MATCH:
                    return dise
            return dise
        else:
            dise = dialog_config.NO_MATCH_BY_RATE
            max_sym_rate = 0.0
            for d in self.dise_sym_num_dict:
                tmp = [v for v in self.dise_sym_num_dict[d].keys() if v in current_slots['inform_slots'].keys()]
                tmp_sum = 0
                cur_dise_sym_sum = 0
                for sym in tmp:
                    tmp_sum += self.dise_sym_num_dict[d][sym]
                for sym in self.dise_sym_num_dict[d]:
                    cur_dise_sym_sum += self.dise_sym_num_dict[d][sym]
                # tmp_rate = float(len(tmp))/float(len(self.req_dise_sym_dict[dise]))
                tmp_rate = float(tmp_sum) / float(cur_dise_sym_sum)
                if tmp_rate > max_sym_rate:
                    max_sym_rate = tmp_rate
                    dise = d
            return dise


