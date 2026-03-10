import os
import sys
import logging
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import is_default_gpu
from utils.logger import print_progress
from utils.data import PickSpecificWords

# 初始化 logger
logger = logging.getLogger(__name__)

class BaseAgent(object):
    ''' Base class for an REVERIE agent to generate and save trajectories. '''

    def __init__(self, env):
        self.env = env
        self.results = {}
        self.feature_size = 768

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'trajectory': v['path'], 'pred_objid': v['pred_objid']})
            if detailed_output:
                output[-1]['details'] = v['details']
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, z_dicts={}, z_front_dict={}, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(z_dicts=z_dicts, z_front_dict=z_front_dict,**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
        else:   # Do a full round
            while True:
                for traj in self.rollout(z_dicts=z_dicts, z_front_dict=z_front_dict,**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                if looped:
                    break

    def test_viz(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
        else:   # Do a full round
            while True:
                for traj in self.rollout_viz(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                if looped:
                    break

class Seq2SeqAgent(BaseAgent):
    env_actions = {
      'left': (0, -1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0, -1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0,tok=None):
        super().__init__(env)
        self.args = args
        self.tok = tok
        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank

        if args.z_instr_update:
            self.word_picker = PickSpecificWords(cat_file=args.cat_file)
            self.instr_specific_dict = defaultdict(lambda:[])
        
        # Models
        self._build_model()

        if self.args.world_size > 1:
            self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
            self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

        self.models = (self.vln_bert, self.critic)
        self.device = torch.device('cuda:%d'%self.rank) 

        # Optimizers
        if self.args.optim == 'rms':
            optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            optimizer = torch.optim.SGD
        else:
            assert False
        if self.default_gpu:
            print('Optimizer: %s' % self.args.optim)

        self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, reduction='sum')

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _build_model(self):
        raise NotImplementedError('child class should implement _build_model: self.vln_bert & self.critic')

    def test(self, use_dropout=False, feedback='argmax', iters=None, z_dicts=None, z_front_dict=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
            
        super().test(iters=iters, z_dicts=z_dicts, z_front_dict=z_front_dict)
    
    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()
    
    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
    
    def accumulate_gradient(self, feedback='teacher', z_dicts={}, z_front_dict={}, **kwargs):
        self.vln_bert.train()
        if self.args.train_alg == 'imitation':
                self.feedback = 'teacher'
                self.rollout(
                    train_ml=1., train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs
                )
        elif self.args.train_alg == 'dagger':
            if self.args.ml_weight != 0:
                self.feedback = 'teacher'
                self.rollout(
                    train_ml=self.args.ml_weight, train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs
                )
            self.feedback = self.args.dagger_sample
            self.rollout(train_ml=1, train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs)
        else:
            assert False

    def train(self, n_iters, feedback='teacher', z_dicts={}, z_front_dict={}, **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        
        self.losses = []
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0

            if self.args.train_alg == 'imitation':
                self.feedback = 'teacher'
                self.rollout(
                    train_ml=1., train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs
                )
            elif self.args.train_alg == 'dagger': 
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(
                        train_ml=self.args.ml_weight, train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs
                    )
                self.feedback = self.args.dagger_sample
                self.rollout(train_ml=1, train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs)
            else:
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(
                        train_ml=self.args.ml_weight, train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs
                    )
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs)

            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()

            if self.args.aug is None:
                print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict()
            }
            if self.args.save_optimizer:
                states[name]['optimizer'] = optimizer.state_dict()
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' 
        Loads parameters with Smart Matching for VGGT/Causal modules.
        Reports Extra/Missing keys accurately based on the final loaded dict.
        '''
        # ==========================================
        # [手动实验开关] 
        # 设置为 False: 刻意抛弃 Checkpoint 中的 vggt 参数，强制随机初始化
        # 设置为 True : 正常加载 Checkpoint 中的 vggt 参数
        # ==========================================
        LOAD_VGGT_EMBEDDING = False
        
        logger.info(f'Load the model from {path} | Manually set LOAD_VGGT_EMBEDDING = {LOAD_VGGT_EMBEDDING}')
        # 兼容 CPU/GPU 加载，防止 OOM
        states = torch.load(path, map_location=self.device)

        # 1. 容器解包：处理嵌套结构
        if 'vln_bert' not in states:
            states = {'vln_bert': states}

        def recover_state(name, model, optimizer):
            if name not in states:
                return

            # 获取当前模型定义的参数名集合
            model_keys = set(model.state_dict().keys())
            
            # 获取 Checkpoint 里的参数字典
            source_state = states[name]
            if 'state_dict' in source_state:
                source_state = source_state['state_dict']

            new_state_dict = {}
            vggt_loaded_count = 0
            vggt_skipped_count = 0

            # ================= [ 核心：智能匹配与筛选逻辑 ] =================
            for k, v in source_state.items():
                
                # [核心拦截] 如果开关关闭，且参数名包含 vggt，直接跳过
                if (not LOAD_VGGT_EMBEDDING) and ('vggt' in k):
                    vggt_skipped_count += 1
                    continue
                
                # A. 彻底清洗前缀，获得“裸名”
                clean_k = k
                if clean_k.startswith('module.'): clean_k = clean_k[7:]
                if clean_k.startswith('vln_bert.'): clean_k = clean_k[9:]
                if clean_k.startswith('bert.'): clean_k = clean_k[5:]

                # B. 寻找宿主：尝试多种前缀组合去匹配当前模型
                target_k = None
                potential_keys = [
                    'vln_bert.' + clean_k,  # 优先级 1: 带 vln_bert.
                    clean_k,                # 优先级 2: 裸名
                    'bert.' + clean_k       # 优先级 3: 带 bert. (兼容旧代码)
                ]

                # C. 匹配判定
                for pk in potential_keys:
                    if pk in model_keys:
                        target_k = pk
                        break
                
                # D. 装载决策
                if target_k:
                    new_state_dict[target_k] = v
                    if 'vggt' in target_k:
                        vggt_loaded_count += 1

            # ================= [ 日志报告：还原 Extra 和 Missing ] =================
            loaded_keys = set(new_state_dict.keys())
            missing_keys = model_keys - loaded_keys
            
            # 重新构建 Checkpoint 的“裸名集合”用于对比 (排除被我们主动跳过的vggt，以免在Extra里报虚假警告)
            source_clean_keys = set()
            for k in source_state.keys():
                if (not LOAD_VGGT_EMBEDDING) and ('vggt' in k):
                    continue # 不计入 extra 统计
                ck = k
                if ck.startswith('module.'): ck = ck[7:]
                if ck.startswith('vln_bert.'): ck = ck[9:]
                if ck.startswith('bert.'): ck = ck[5:]
                source_clean_keys.add(ck)
            
            model_clean_keys = set()
            for k in model_keys:
                ck = k
                if ck.startswith('vln_bert.'): ck = ck[9:]
                if ck.startswith('bert.'): ck = ck[5:]
                model_clean_keys.add(ck)

            extra_clean = source_clean_keys - model_clean_keys
            
            # === 打印 VGGT 状态 ===
            if LOAD_VGGT_EMBEDDING:
                print(f"\n[Load Status] 成功匹配并加载 VGGT 参数: {vggt_loaded_count} 个")
            else:
                print(f"\n[Load Status] ⚠️ 主动跳过 VGGT 参数加载: {vggt_skipped_count} 个 (使用随机初始化)")
            
            # === 还原日志打印逻辑 ===
            if len(extra_clean) > 0:
                relevant_extra = [k for k in list(extra_clean) if not k.startswith('tim_') and not k.startswith('mlm_')]
                if len(relevant_extra) > 0:
                    print(f'!!! WARNING !!! Real Extra keys (Ignored): {sorted(relevant_extra)[:]} ... Total: {len(relevant_extra)}')

            if len(missing_keys) > 0:
                print(f'!!! WARNING !!! Missing keys in model (Random Init): {sorted(list(missing_keys))[:]} ... Total: {len(missing_keys)}')
                
                vggt_in_missing = [k for k in missing_keys if 'vggt' in k]
                if vggt_in_missing:
                    print(f"❌ 警告: VGGT 参数当前处于缺失(随机初始化)状态: {len(vggt_in_missing)} 个")
                else:
                    print("✅ 确认: VGGT 参数不在缺失列表中 (已正确加载)。")

            print("="*50 + "\n")

            # E. 执行最终加载
            model.load_state_dict(new_state_dict, strict=False)

            if getattr(self.args, 'resume_optimizer', False) and 'optimizer' in states[name]:
                try:
                    optimizer.load_state_dict(states[name]['optimizer'])
                    print("✅ 成功加载 optimizer state.")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")

        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer)]
        for param in all_tuple:
            recover_state(*param)

        return states['vln_bert'].get('epoch', 0) - 1