import copy
import torch
from torch import nn

import sys
sys.path.append('latent_safety')

import dreamerv3_torch.networks_mile as networks
import dreamerv3_torch.tools as tools
import dreamerv3_torch.utils as utils
from dreamerv3_torch.uncertainty import OneStepPredictor

to_np = lambda x: x.detach().cpu().numpy()




class Policy(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(True),
            nn.Linear(in_channels // 2, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.fc(x)


class WorldModelMile(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModelMile, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSMMile(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["action"] = networks.MLP(
            feat_size,
            (5, ),   # 连续动作就输出动作维度
            config.action_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.action_head["dist"],   # 例如 normal / mse / trunc_normal 等
            outscale=config.action_head["outscale"],
            device=config.device,
            name="Action",
        )

        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            action=config.action_head["loss_scale"],
        )

    def _train(self, data, ensemble: OneStepPredictor| None = None):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.amp.autocast('cuda',enabled = self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"], policy=self.heads["action"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.amp.autocast('cuda',enabled = self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )

        # Disagreement Ensemble Training!
        if ensemble is not None:
            with tools.RequiresGrad(ensemble):
                stoch = post["stoch"]
                if self._config.dyn_discrete:
                    stoch = torch.reshape(
                        stoch, (stoch.shape[:-2] + ((stoch.shape[-2] * stoch.shape[-1]),))
                    )
                target = {
                    "embed": embed,
                    "stoch": stoch,
                    "deter": post["deter"],
                    "feat": feat,
                }[self._config.disag_target]
                with torch.no_grad():
                    inputs = self.dynamics.get_feat(post)
                
                ensemble_mets = ensemble.train_ensemble_penn_fixed(inputs, data["action"], target, data["is_first"])
                metrics.update({k: v for k, v in ensemble_mets.items()})

        post = {k: v.detach() for k, v in post.items()}

        return post, context, metrics

    def train_uncertainty_only(self, data, ensemble: OneStepPredictor| None = None):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with torch.no_grad() :
            embed = self.encoder(data)
            post, prior = self.dynamics.observe(
                embed, data["action"], data["is_first"], policy=self.heads["action"]
            )

        # Disagreement Ensemble Training!
        if ensemble is not None:
            with tools.RequiresGrad(ensemble):
                stoch = post["stoch"]
                if self._config.dyn_discrete:
                    stoch = torch.reshape(
                        stoch, (stoch.shape[:-2] + ((stoch.shape[-2] * stoch.shape[-1]),))
                    )
                target = {
                    "embed": embed,
                    "stoch": stoch,
                    "deter": post["deter"],
                }[self._config.disag_target]
                with torch.no_grad():
                    inputs = self.dynamics.get_feat(post)
                
                #  Finetuning ensemble!
                for _ in range(1): #range(5):
                    ensemble_mets = ensemble.train_ensemble_penn_fixed(inputs, data["action"], target, data["is_first"])
                metrics = ensemble_mets


        post = {k: v.detach() for k, v in post.items()}

        return metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        
        for k in obs.keys():        
            if "cam" in k:
                obs[k] = obs[k] / 255.0

        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)

        # FIXME: For failure -> (N,T,1)
        obs["failure"] = obs["failure"].unsqueeze(-1)

        return obs

    def video_pred(self, data, ensemble: OneStepPredictor| None = None):
        data = self.preprocess(data)
        embed = self.encoder(data)

        obs_step = 5

        states, _ = self.dynamics.observe(
            embed[:6, :obs_step], data["action"][:6, :obs_step], data["is_first"][:6, :obs_step]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, obs_step:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        truth = data["image"][:6]

        # Clip when finished
        row, col = torch.where(data['is_first'][:6, obs_step:] == 1.)
        for i in range(row.size(0)):
            data['is_first'][row[i], obs_step+col[i]:] = 1.
            openl[row[i], col[i]:] = openl[row[i], col[i]-1]
            truth[row[i], obs_step+col[i]:] = truth[row[i], obs_step+col[i]-1]

        # observed image is given until 5 steps
        model = torch.cat([recon[:, :obs_step], openl], 1)
        error = (model - truth + 1.0) / 2.0

        video_pred = torch.cat([truth, model, error], 2)

        # Also visualize uncertainties
        if ensemble is not None:
            with torch.no_grad():
                feat_post = self.dynamics.get_feat(states)
                feat_prior = self.dynamics.get_feat(prior)
                feat = torch.cat([feat_post, feat_prior], 1)
                # (feat: s_t, action: a_t-1). The last image have no action! -> cannot measure uncertainty.
                # action = data["action"][:6]  # FIXME -> action is "prev_action."
                action = torch.roll(data["action"][:6], shifts=-1, dims=1)
                inputs = torch.concat([feat, action], -1)
                #penn
                disagreement_ensemble = ensemble.intrinsic_reward_penn(inputs)
                
            
            video_pred = utils.concat_uncertainty_with_video(data, video_pred, disagreement_ensemble)

        return video_pred
