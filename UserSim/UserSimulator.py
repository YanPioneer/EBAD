import random
import dialog_config


class UserSimulator:
    """ user goal  """

    def __init__(self, sym_dict=None, goals_set=None, params=None):

        self.sym_dict = sym_dict  # all_symptom
        self.goals_set = goals_set  # 所有用户目标

        self.max_turn = params['max_turn']
        self.use_mode = params['use_mode']
        self.ture_hit = 0  # 只记录患者存在的症状命中与否，利于剪枝，给5奖励值？
        self.repeat = 0
        self.turn_reward = 0

        self.test = 0
        self.test_goal = []

        self.state = {}
        self.include_sym = []

        self.episode = 0

    def reset(self):
        """ choose a goal """
        self.include_sym = []
        dialog_config.include_sym_is_empty = False

        self.state = {}  # 存储信息
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = 0

        self.episode_over = False  # 表征回合是否结束
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        self.goal = self._sample_goal()

        # sample first action
        user_action = self.start_action()
        if len(self.include_sym) == 0:
            dialog_config.include_sym_is_empty = True

        self.episode += 1

        return user_action, self.goal

    def start_action(self):
        self.state['diaact'] = "request"  # 初始动作，告知显性症状
        self.state['request_slots']['disease'] = 'UNK'

        if len(self.goal['explicit_inform_slots']) > 0:
            for slot in self.goal['explicit_inform_slots']:
                if self.goal['explicit_inform_slots'][slot] == True or self.goal['explicit_inform_slots'][slot] == '1':
                    self.state['inform_slots'][slot] = dialog_config.TRUE
                if self.goal['explicit_inform_slots'][slot] == False or self.goal['explicit_inform_slots'][slot] == '0':
                    self.state['inform_slots'][slot] = dialog_config.FALSE

        for s in list(self.goal['implicit_inform_slots'].keys()):
            self.include_sym.append(s)

        start_action = {}
        start_action['diaact'] = self.state['diaact']
        start_action['request_slots'] = self.state['request_slots']
        start_action['inform_slots'] = self.state['inform_slots']
        start_action['turn'] = self.state['turn']

        return start_action

    def step(self, system_action):
        """ transition """
        self.ture_hit = 0
        self.turn_reward = 0
        self.state['turn'] += 2

        sys_act = system_action['diaact']

        if 0 < self.max_turn < self.state['turn']:
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.episode_over = True
            self.state['diaact'] = "closing"
            self.turn_reward += -1 * self.max_turn
        else:
            self.state['history_slots'].update(self.state['inform_slots'])  # add inform slot to history
            self.state['inform_slots'].clear()

            if sys_act == 'inform':
                self.response_inform(system_action)
            elif sys_act == 'request':
                self.response_request(system_action)
            elif sys_act == 'thanks':
                self.response_thanks(system_action)

        # 添加噪声, 未作
        self.corrupt()

        response_action = {}
        response_action['diaact'] = self.state['diaact']
        response_action['inform_slots'] = self.state['inform_slots']
        response_action['request_slots'] = self.state['request_slots']
        response_action['turn'] = self.state['turn']

        if len(self.include_sym) == 0:
            dialog_config.include_sym_is_empty = True

        return response_action, self.turn_reward, self.episode_over, self.dialog_status, self.ture_hit

    def response_inform(self, system_action):
        """ diagnosis result """
        self.episode_over = True
        self.dialog_status = dialog_config.SUCCESS_DIALOG
        # error result
        self.state['request_slots']['disease'] = system_action['inform_slots']['disease']
        if self.state['request_slots']['disease'] == 'UNK' or self.state['request_slots']['disease'] != self.goal['disease_tag']:
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.turn_reward += -1 * self.max_turn
        else:
            self.turn_reward += 1 * self.max_turn  # v1为2
        self.state['diaact'] = 'thanks'

    def response_request(self, system_action):
        """ inquiry implicit symptom """
        if len(system_action['request_slots'].keys()) > 0:  # 目前一轮设定一个疾病
            slot = list(system_action['request_slots'].keys())[0]
            if slot in self.goal['implicit_inform_slots'].keys():
                self.include_sym.remove(slot)  # 删除

                if self.goal['implicit_inform_slots'][slot] == True or self.goal['implicit_inform_slots'][slot] == '1':
                    self.state['diaact'] = "confirm"
                    self.state['inform_slots'][slot] = dialog_config.TRUE
                    self.ture_hit = 1
                    self.turn_reward += 5
                elif self.goal['implicit_inform_slots'][slot] == False or self.goal['implicit_inform_slots'][slot] == '0':
                    self.state['diaact'] = "deny"
                    self.state['inform_slots'][slot] = dialog_config.FALSE
                    self.ture_hit = 1
                    self.turn_reward += 1
            else:
                self.state['diaact'] = "not_sure"
                self.state['inform_slots'][slot] = dialog_config.NOT_SURE
                self.turn_reward -= 1

    def response_thanks(self, system_action):  # 会有??
        self.episode_over = True
        self.dialog_status = dialog_config.SUCCESS_DIALOG
        if self.state['request_slots']['disease'] == 'UNK' or self.state['request_slots']['disease'] != self.goal['disease_tag']:
            self.dialog_status = dialog_config.FAILED_DIALOG
        self.state['diaact'] = "closing"

    def _sample_goal(self):
        """ sample use goal """
        if self.test == 1:
            sample_goal = random.choice(self.test_goal)
            # sample_goal = self.goals_set[self.use_mode][self.episode]
            self.test_goal.remove(sample_goal)
        else:
            sample_goal = random.choice(self.goals_set[self.use_mode])
            # sample_goal = self.goals_set[self.use_mode][0]
        return sample_goal

    def corrupt(self):
        pass

    def calculate_reward(self):  # 互信息计算奖励值
        pass
