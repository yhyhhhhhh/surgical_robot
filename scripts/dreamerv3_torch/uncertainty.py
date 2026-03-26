import torch
from torch import nn
from torch import distributions as torchd

import sys
sys.path.append('latent_safety')
import dreamerv3_torch.tools as tools
from dreamerv3_torch.ensemble.penn import EnsembleStochasticLinear 

def clamp_preserve_gradients(x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """
    Clamps the values of the tensor into ``[lower, upper]`` but keeps the gradients.

    Args:
        x: The tensor whose values to constrain.
        lower: The lower limit for the values.
        upper: The upper limit for the values.

    Returns:
        The clamped tensor.
    """
    return x + (x.clamp(min=lower, max=upper) - x).detach()
import torch
import torch.nn as nn
# 假设存在 tools 模块，通常在 Dreamer 算法库中包含优化器等工具类

class OneStepPredictor(nn.Module):
    def __init__(self, config, world_model):
        super(OneStepPredictor, self).__init__()
        # 保存传入的配置参数
        self._config = config
        # 判断是否使用 AMP (自动混合精度) 加速训练，如果配置精度为 16 则是，否则否
        self._use_amp = True if config.precision == 16 else False
        
        # 根据世界模型（World Model）是离散动态还是连续动态来计算输入特征的维度大小
        if config.dyn_discrete:
            # 离散情况：特征大小 = 随机状态类别数 * 离散维度 + 确定性状态维度
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            stoch = config.dyn_stoch * config.dyn_discrete
        else:
            # 连续情况：特征大小 = 随机状态维度 + 确定性状态维度
            feat_size = config.dyn_stoch + config.dyn_deter
            stoch = config.dyn_stoch
            
        # 根据配置决定预测的目标 (target) 是什么，并设定对应的输出维度
        size = {
            "embed": world_model.embed_size, # 预测图像嵌入向量
            "stoch": stoch,                  # 预测随机状态部分
            "deter": config.dyn_deter,       # 预测确定性状态部分
            "feat": config.dyn_stoch + config.dyn_deter, # 预测完整特征
        }[self._config.disag_target]

        # 计算集成网络的输入维度：特征大小 + 动作维度（如果配置了动作条件）
        input_dim = feat_size + (config.num_actions if config.disag_action_cond else 0)

        # 初始化集成随机线性网络（用于估计认知不确定性）
        self._networks = EnsembleStochasticLinear(in_features=input_dim, 
                                                 out_features=size,
                                                 hidden_features=input_dim,
                                                 ensemble_size=config.disag_models, # 集成模型的数量（例如 5 或 10）
                                                 explore_var='jrd',                 # 探索方差的计算方式
                                                 residual=True)                     # 使用残差连接
        
        # 设置损失函数为自定义的高斯负对数似然损失
        self.criterion = self.gaussian_nll_loss 
        
        # 配置优化器的参数字典
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        # 初始化自定义的优化器工具类
        self._expl_opt = tools.Optimizer(
            "ensemble",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw,
        )
        self.config = config

    def gaussian_nll_loss(self, mu, target, var):
        # 自定义高斯负对数似然损失 (Gaussian Negative Log Likelihood Loss)
        # 数学公式: 0.5 * (log(var) + (target - mu)^2 / var)
        loss = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
        return torch.mean(loss) # 返回所有样本损失的均值
    
    def intrinsic_reward_penn(self, inputs):
        # 计算基于 PENN (Probabilistic Ensemble Neural Network) 的内部奖励
        self._networks.eval() # 将网络设置为评估模式

        # 检查输入是否为 3D 张量：(Batch_size, Time_steps, Dimension)
        if len(inputs.shape) == 3:
            N, T, D = inputs.shape
            # 展平前两个维度以便输入网络：(N*T, D)
            inputs = inputs.reshape(N * T, D)

            with torch.no_grad(): # 计算奖励不需要梯度
                ensemble_outputs = self._networks(inputs)
                div = ensemble_outputs[-1] # 获取集成输出的最后一项，通常是方差或散度(divergence)
            
            # 恢复到原来的形状 (N, T, 散度维度)
            div = div.view(N, T, -1)
        else:
            # 如果是 2D 张量，直接前向传播
            with torch.no_grad():
                ensemble_outputs = self._networks(inputs)
                div = ensemble_outputs[-1]

        # 如果配置项要求，对散度取对数处理
        if self._config.disag_log:
            div = torch.log1p(div)

        return div # 返回散度作为内部探索奖励
    
    def train_ensemble_penn_fixed(self, feats, actions, targets, is_first):
        # 训练这个集成网络的方法
        self._networks.train() # 切换到训练模式
        with torch.cuda.amp.autocast(self._use_amp): # 开启/关闭自动混合精度

            # 时间步对齐：用 t 时刻的特征和 t+1 时刻的动作去预测 t+1 时刻的目标
            feats = feats[:, :-1]   # 去掉最后一个时间步：(N, T-1)
            actions = actions[:, :-1] # 去掉第一个时间步：(N, T-1)
            # 拼接特征和动作作为输入
            inputs = torch.concat([feats, actions], -1) 
            targets = targets[:, 1:] # 目标为 t+1 时刻的值：(N, T-1)

            # 过滤掉无效的时间步（例如：跨越了回合边界的转移）
            # 如果 t+1 时刻是新回合的第一步 (is_first==1)，那么 t 到 t+1 的转移是无效的
            valid_idx = torch.roll(is_first, shifts=-1, dims=1)[:, :-1] == 0.

            # 提取有效的输入和目标
            valid_inputs = inputs[valid_idx]
            valid_targets = targets[valid_idx]

            # 分离计算图 (detach)：因为我们不希望训练集成网络时，梯度回传去改变世界模型（World Model）
            valid_inputs = valid_inputs.detach()
            valid_targets = valid_targets.detach()
            
            train_loss = torch.zeros(1, device=valid_inputs.device)
            
            # 遍历集成模型中的每一个独立的网络进行训练
            for i in range(self.config.disag_models):                
                # 单独前向传播第 i 个模型，获取预测的均值(mu)和对数标准差(log_std)
                (mu, log_std) = self._networks.single_forward(
                    valid_inputs, index=i)

                yhat_mu = mu.unsqueeze(0) # 增加一个维度以匹配目标格式
                # 计算方差 var = (exp(log_std))^2
                var = torch.square(torch.exp(log_std.unsqueeze(0))).clamp_min(1e-6)
                # 计算高斯 NLL 损失
                loss = self.gaussian_nll_loss(yhat_mu, valid_targets, var)
                loss = loss.mean()
                # 使用自定义优化器更新网络参数
                self._expl_opt(loss, self._networks.parameters())
                
                train_loss += loss # 累加所有模型的损失

        # 记录训练指标：平均每个模型的损失
        metrics = {"ensemble_loss": train_loss.item() / self.config.disag_models}

        # 评估训练后的网络分歧程度（散度），记录在日志中
        with torch.no_grad():
            div = self.intrinsic_reward_penn(valid_inputs).mean()
        metrics["log_disagreement"] = div.cpu().numpy()

        return metrics # 返回指标字典
    
class DensityEstimator_MAF(nn.Module):
    def __init__(self, config):
        super(DensityEstimator_MAF, self).__init__()
        self.config = config
        input_dim = config.dyn_deter
        self._use_amp = True if config.precision == 16 else False
        self._networks = MaskedAutoregressiveFlow(dim=input_dim, num_layers=4, hidden_layer_size=input_dim)
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "nf_density",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw,
        )
        self.norm_max = 30

    def train_density_estimator(self, x):
        x = x.detach()
        N, T, D = x.shape
        x = x.reshape(-1, D)

        log_prob = self._networks.forward(x)
        # log_prob = clamp_preserve_gradients(log_prob, lower=-30, upper=30)
        loss = -torch.mean(log_prob)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            self._expl_opt(loss, self._networks.parameters())

        with torch.no_grad():
            prob = torch.exp(log_prob).view(N, T)
            prob[torch.isinf(prob)] = self.norm_max
            prob[torch.isnan(prob)] = 0
            density = prob

        metrics = {"density_loss": loss.item(), "density": density.mean().item()}

        return metrics
    
    def calculate_likelihood(self, x):

        self._networks.eval()
        N, T, D = x.shape
        x = x.reshape(-1, D)
        log_prob = self._networks.forward(x)
        prob = torch.exp(log_prob).view(N, T)
        prob[torch.isnan(prob)] = 0
        prob = torch.clamp(prob, min=0, max=self.norm_max)
        self._networks.train()

        return prob


class OneStepPredictorUnitVariance(nn.Module):
    def __init__(self, config, world_model):
        super(OneStepPredictorUnitVariance, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            stoch = config.dyn_stoch * config.dyn_discrete
            dist = "onehot"
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            stoch = config.dyn_stoch
            dist = "symlog_mse" #"normal_std_fixed"
        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.dyn_deter,
            "feat": config.dyn_stoch + config.dyn_deter,
        }[self._config.disag_target]
        kw = dict(
            inp_dim=feat_size
            + (
                config.num_actions if config.disag_action_cond else 0
            ),  # pytorch version
            dist = dist, # Normal.
            shape=size,
            layers=config.disag_layers,
            units=config.disag_units,
            act=config.act,
        )

        input_dim = feat_size + (config.num_actions if config.disag_action_cond else 0)

        self._networks = EnsembleStochasticLinearUnitVariance(in_features=input_dim, 
                                                 out_features=size,
                                                 hidden_features=input_dim, #hidden_features=config.disag_units, #
                                                 ensemble_size=config.disag_models,
                                                 explore_var='jrd', 
                                                 residual=True)
        
        torch.backends.cudnn.benchmark = True
        
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "explorer",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw,
        )
        self.config = config
    
    def intrinsic_reward_penn(self, inputs):

        self._networks.eval()

        if len(inputs.shape) == 3:
            N, T, D = inputs.shape
            inputs = inputs.reshape(N * T, D)

            with torch.no_grad():
                ensemble_outputs = self._networks(inputs)
                div = ensemble_outputs[-1]
            
            div = div.view(N, T, -1)
        else:
            with torch.no_grad():
                ensemble_outputs = self._networks(inputs)
                div = ensemble_outputs[-1]

        if self._config.disag_log:
            div = torch.log(div)

        return div
    
    def train_ensemble_penn_fixed(self, feats, actions, targets, is_first):
        self._networks.train()
        with torch.cuda.amp.autocast(self._use_amp):
            feats = feats[:, :-1] # N, T-1
            actions = actions[:, 1:] # N, T-1
            inputs = torch.concat([feats, actions], -1)
            targets = targets[:, 1:] # N, T-1

            valid_idx = torch.roll(is_first, shifts=-1, dims=1)[:, :-1] == 0.

            valid_inputs = inputs[valid_idx]
            valid_targets = targets[valid_idx]

            valid_inputs = valid_inputs.detach()
            valid_targets = valid_targets.detach()
            
            train_loss = torch.FloatTensor([0]).cuda()

            for i in range(self.config.disag_models):                
                mu = self._networks.single_forward(
                    valid_inputs, index=i)
                
                yhat_mu = mu.unsqueeze(0)
                loss = (yhat_mu - valid_targets).pow(2)
                loss = loss.mean()
                self._expl_opt(loss, self._networks.parameters())
                train_loss += loss
            
        metrics = {"explorer_loss": train_loss.item() / self.config.disag_models}

        with torch.no_grad():
            div = self.intrinsic_reward_penn(valid_inputs).mean()
        metrics["log_disagreement"] = div.cpu().numpy()

        return metrics
    



import normflows as nf
class DensityEstimator(nn.Module): 

    def __init__(self, config):
        super(DensityEstimator, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False

        feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        input_dim = feat_size     
        self.config = config

        # Set base distribuiton
        self.q0 = nf.distributions.DiagGaussian(input_dim, trainable=True)
        # self.q0 = nf.distributions.DiagGaussian(input_dim, trainable=False)
        flows = [
            nf.flows.CoupledRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=2, num_hidden_channels=input_dim*2),
            nf.flows.LULinearPermute(input_dim),
            nf.flows.CoupledRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=2, num_hidden_channels=input_dim*2),
            nf.flows.LULinearPermute(input_dim),
            nf.flows.CoupledRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=2, num_hidden_channels=input_dim*2),
            nf.flows.LULinearPermute(input_dim),
            nf.flows.CoupledRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=2, num_hidden_channels=input_dim*2),
            nf.flows.LULinearPermute(input_dim)
            ]

        self._networks = nf.NormalizingFlow(q0=self.q0, flows=flows)
        self._networks = self._networks.cuda()

        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "nf_density",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw,
        )


    def train_density_estimator(self, x):
        x = x.detach()
        N, T, D = x.shape
        x = x.view(-1, D)

        torch.use_deterministic_algorithms(False)
        # import pdb; pdb.set_trace()
        loss = self._networks.forward_kld(x)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            self._expl_opt(loss, self._networks.parameters())
        torch.use_deterministic_algorithms(True)
        metrics = {"density_loss": loss.item()}

        return metrics
    
    def calculate_likelihood(self, x):

        self._networks.eval()

        if len(x.shape) == 3:
            N, T, D = x.shape
            x = x.view(-1, D)
            torch.use_deterministic_algorithms(False)
            log_prob = self._networks.log_prob(x)
            torch.use_deterministic_algorithms(True)
            prob = torch.exp(log_prob).view(N, T)
            prob = torch.clamp(prob, min=0, max=1)
            self._networks.train()
        
        else:
            torch.use_deterministic_algorithms(False)
            log_prob = self._networks.log_prob(x)
            torch.use_deterministic_algorithms(True)
            prob = torch.exp(log_prob)
            prob = torch.clamp(prob, min=0, max=1)
            self._networks.train()


        return prob