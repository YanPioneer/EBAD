import copy

import torch

from dialog_system.state_tracker import StateTracker
import dialog_config
from torch.distributions import Categorical


class DialogManager:

    def __init__(self, agent, user, kgraph):
        self.agent = agent
        self.user = user
        self.kgraph = kgraph

        # 用于计算互信息，记录上次和本次疾病分布
        self.disease_dis = torch.zeros(self.agent.batch_size, self.agent.dis_shape)
        self.current_disease_dis = torch.zeros(self.agent.batch_size, self.agent.dis_shape)

        self.state_tracker = StateTracker()
        self.user_action = None
        self.reward = 0
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET
        self.hit_rate = 0

    def initialize_episode(self):
        """ start a new episode """
        # print("New Episode!")
        self.disease_mask = copy.deepcopy(dialog_config.disease_mask).to(dialog_config.device).reshape(-1, 27)
        self.symptoms_mask = copy.deepcopy(dialog_config.symptoms_mask).to(dialog_config.device).reshape(-1, 358)
        self.reward = 0
        self.episode_over = False
        self.state_tracker.initialize_episode()  # 对话状态追踪
        self.kgraph.initialize_adj()  # 初始化邻接矩阵, 自动初始化
        self.initial_kg_matrix = copy.deepcopy(self.kgraph.kg_matrix).view(-1, self.agent.entity_num, self.agent.entity_num)
        self.kg_matrix = copy.deepcopy(self.initial_kg_matrix)
        self.user_action, self.goal = self.user.reset()
        self.state_tracker.update(user_action=self.user_action)  # 患者回答时剪枝
        self.kgraph.update_adj(self.state_tracker.current_confirm_sym)  # 修改图谱的邻接矩阵，根据患者自述剪枝
        self.next_kg_matrix = copy.deepcopy(self.kgraph.kg_matrix).view(-1, self.agent.entity_num, self.agent.entity_num)

        self.hit_rate = 0
        self.agent.initialize_episode()

        ############# record processing information #################
        if self.agent.warm_start == 2:
            if self.agent.mode == 'train':
                with open(dialog_config.diagnosis_re, 'a+', encoding='utf-8') as ref:
                    ref.write('\nNew Episode!\n')
                    ref.write('Sample a goal:' + str(self.goal) + '\n')
                    ref.write('User action:' + str(self.user_action) + '\n')
            elif self.agent.mode == 'test':
                with open(dialog_config.test_re, 'a+', encoding='utf-8') as ref:
                    ref.write('\nNew Episode!\n')
                    ref.write('Sample a goal:' + str(self.goal) + '\n')
                    ref.write('User action:' + str(self.user_action) + '\n')

    def next_dialog(self, record_training_data=True):
        self.kg_matrix = copy.deepcopy(self.kgraph.kg_matrix).view(-1, self.agent.entity_num, self.agent.entity_num)
        # self.disease_dis = self.current_disease_dis
        for i in range(self.kg_matrix.size(0)):
            for j in range(self.agent.dis_shape):
                if torch.equal(self.kg_matrix[i, j, :], torch.zeros(self.kg_matrix[i, j, :].size()).to(dialog_config.device)):
                    self.disease_mask[0][j] = 0.
            for k in range(self.agent.sym_shape):
                if torch.equal(self.kg_matrix[i, k + self.agent.dis_shape, :], torch.zeros(self.kg_matrix[i, k + self.agent.dis_shape, :].size()).to(dialog_config.device)):
                    self.symptoms_mask[0][k] = 0.
        disease_mask = copy.deepcopy(self.disease_mask)
        symptoms_mask = copy.deepcopy(self.symptoms_mask)
        # 剪枝在此处
        self.dialog_context = self.state_tracker.get_state_for_agent()  # 将上下文告知agent
        self.agent_action, self.disease_dis = self.agent.state_to_action(self.kg_matrix, self.dialog_context, disease_mask, symptoms_mask)

        ############# record processing information #################
        if self.agent.warm_start == 2:
            if self.agent.mode == 'train':
                with open(dialog_config.diagnosis_re, 'a+', encoding='utf-8') as ref:
                    ref.write('Agent action:' + str(self.agent_action) + '\n')
            elif self.agent.mode == 'test':
                with open(dialog_config.test_re, 'a+', encoding='utf-8') as ref:
                    ref.write('Agent action:' + str(self.agent_action) + '\n')

        # 更新记录的动作
        self.state_tracker.update(agent_action=self.agent_action)

        # next step
        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
        self.user_action, self.reward, self.episode_over, self.dialog_status, hit = self.user.step(self.sys_action)
        self.hit_rate += hit

        ############# record processing information #################
        if self.agent.warm_start == 2:
            if self.agent.mode == 'train':
                with open(dialog_config.diagnosis_re, 'a+', encoding='utf-8') as ref:
                    ref.write('User action:' + str(self.user_action) + '\n')
            elif self.agent.mode == 'test':
                with open(dialog_config.test_re, 'a+', encoding='utf-8') as ref:
                    ref.write('User action:' + str(self.user_action) + '\n')

        # if self.episode_over != True:  # 不应加判断
        self.state_tracker.update(user_action=self.user_action)
        # 更新图谱邻接矩阵
        self.kgraph.update_adj(self.state_tracker.current_confirm_sym)
        self.next_kg_matrix = copy.deepcopy(self.kgraph.kg_matrix).view(-1, self.agent.entity_num, self.agent.entity_num)

        for i in range(self.next_kg_matrix.size(0)):
            for j in range(self.agent.dis_shape):
                if torch.equal(self.next_kg_matrix[i, j, :], torch.zeros(self.next_kg_matrix[i, j, :].size()).to(dialog_config.device)):
                    self.disease_mask[0][j] = 0.
            for k in range(self.agent.sym_shape):
                if torch.equal(self.next_kg_matrix[i, k + self.agent.dis_shape, :], torch.zeros(self.next_kg_matrix[i, k + self.agent.dis_shape, :].size()).to(dialog_config.device)):
                    self.symptoms_mask[0][k] = 0.
        next_disease_mask = copy.deepcopy(self.disease_mask)
        next_symptoms_mask = copy.deepcopy(self.symptoms_mask)

        mutual_reward = self.calculate_reward(next_disease_mask, next_symptoms_mask)  # 添加互信息奖励
        self.reward += mutual_reward

        ############# record processing information #################
        if self.agent.warm_start == 2:
            if self.agent.mode == 'train':
                with open(dialog_config.diagnosis_re, 'a+', encoding='utf-8') as ref:
                    ref.write('Mutual Reward:' + str(mutual_reward) + 'Reward:' + str(self.reward) + '\n')
            elif self.agent.mode == 'test':
                with open(dialog_config.test_re, 'a+', encoding='utf-8') as ref:
                    ref.write('Mutual Reward:' + str(mutual_reward) + 'Reward:' + str(self.reward) + '\n')

        #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
        if record_training_data:
            self.agent.register_experience_replay_tuple(self.dialog_context, self.kg_matrix, disease_mask,
                                                        symptoms_mask, self.agent_action['act_slot_response'],
                                                        self.state_tracker.get_state_for_agent(),
                                                        self.next_kg_matrix, next_disease_mask, next_symptoms_mask,
                                                        self.reward, self.episode_over)

        return self.episode_over, self.reward, self.dialog_status, self.hit_rate

    def calculate_reward(self, dis_mask, sym_mask):  # 互信息计算奖励值

        # self.dialog_context = self.state_tracker.get_state_for_agent()  # 将上下文告知agent
        self.agent.disc = True
        _, self.current_disease_dis = self.agent.state_to_action(self.next_kg_matrix, self.state_tracker.get_state_for_agent(), dis_mask, sym_mask)
        self.agent.disc = False
        # print(self.goal)
        # print(self.current_disease_dis, self.disease_dis)
        # print(self.current_disease_dis, self.disease_dis)
        disease_dis = Categorical(self.disease_dis)
        next_disease_dis = Categorical(self.current_disease_dis)

        mutual_info = disease_dis.entropy() - next_disease_dis.entropy()

        return mutual_info * 10.




