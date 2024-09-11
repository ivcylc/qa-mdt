import torch
import torch.nn as nn
import pytorch_lightning as pl
from audioldm_train.utilities.model_util import (
    exists,
    default,
    mean_flat,
    count_params,
    instantiate_from_config,
)

from transformers import GPT2Config, GPT2Model
import torch.optim.lr_scheduler as lr_scheduler


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128], dropout_rate=0.5):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs


class CLAP2AudioMAE(pl.LightningModule):
    def __init__(
        self,
        sequence_gen_length,
        base_learning_rate,
        cond_stage_config,
        use_audiomae_linear=False,
        **kwargs
    ):

        super().__init__()
        assert use_audiomae_linear == False
        self.learning_rate = base_learning_rate
        self.cond_stage_config = cond_stage_config
        self.use_audiomae_linear = use_audiomae_linear

        self.mae_token_num = sequence_gen_length  # 4*4 pooling of the audiomae latent

        self.cond_stage_models = nn.ModuleList([])
        self.instantiate_cond_stage(cond_stage_config)

        self.model = GPT2Model.from_pretrained("gpt2")

        self.linear_clap = nn.Linear(512, 768)

        if use_audiomae_linear:
            # self.linear_audiomae = nn.Linear(768, 768) # TODO remove linear_audiomae
            self.linear_audiomae = None  # TODO remove linear_audiomae

        self.loss_fn = nn.MSELoss()

        self.logger_save_dir = None
        self.logger_exp_name = None
        self.logger_exp_group_name = None
        self.logger_version = None

    def set_log_dir(self, save_dir, exp_group_name, exp_name):
        self.logger_save_dir = save_dir
        self.logger_exp_group_name = exp_group_name
        self.logger_exp_name = exp_name

    def cfg_uncond(self, batch_size):
        unconditional_conditioning = {}
        for key in self.cond_stage_model_metadata:
            model_idx = self.cond_stage_model_metadata[key]["model_idx"]
            unconditional_conditioning[key] = self.cond_stage_models[
                model_idx
            ].get_unconditional_condition(batch_size)
        assert (
            "crossattn_audiomae_pooled" in unconditional_conditioning.keys()
        ), "The module is not initialized with AudioMAE"
        unconditional_conditioning[
            "crossattn_clap_to_audiomae_feature"
        ] = unconditional_conditioning["crossattn_audiomae_pooled"]
        return unconditional_conditioning

    def configure_optimizers(self):
        lr = float(self.learning_rate)
        params = list(self.model.parameters()) + list(self.linear_clap.parameters())

        if self.use_audiomae_linear:
            params += list(self.linear_audiomae.parameters())

        opt = torch.optim.AdamW(params, lr=lr)
        scheduler = lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
        return [opt], [scheduler]

    def training_step(self, batch, batch_idx=None, cond_dict=None):
        if cond_dict is None:
            cond_dict = self.get_input(batch)

        input_embeds, target_embeds = (
            cond_dict["film_clap_cond1"],
            cond_dict["crossattn_audiomae_pooled"][0],
        )

        # Some times if the pooling factor is random, the length of crossattn_audiomae_pooled is not necessary 32, so need to calculate separately
        if "crossattn_audiomae_pooled_44" in cond_dict.keys():
            target_embeds = cond_dict["crossattn_audiomae_pooled_44"][0]

        if self.use_audiomae_linear:
            input_embeds = torch.cat(
                [self.linear_clap(input_embeds), self.linear_audiomae(target_embeds)],
                dim=1,
            )
        else:
            input_embeds = torch.cat(
                [self.linear_clap(input_embeds), target_embeds], dim=1
            )

        output_embeds = self.model(inputs_embeds=input_embeds)["last_hidden_state"]

        target = target_embeds
        output = output_embeds[:, :-1]

        loss = self.loss_fn(output, target)

        self.log(
            "train/loss_clap_2_audiomae",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        self.log(
            "global_step_audiomae",
            float(self.global_step),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return loss

    def generate(self, batch, cond_dict=None, no_grad=False):
        if cond_dict is None:
            cond_dict = self.get_input(batch)
        input_embeds = cond_dict["film_clap_cond1"]
        steps = self.mae_token_num

        if no_grad:
            with torch.no_grad():
                model_input = self.linear_clap(input_embeds)
                for _ in range(steps):
                    output = self.model(inputs_embeds=model_input)["last_hidden_state"]
                    model_input = torch.cat([model_input, output[:, -1:, :]], dim=1)
        else:
            model_input = self.linear_clap(input_embeds)
            for _ in range(steps):
                output = self.model(inputs_embeds=model_input)["last_hidden_state"]
                model_input = torch.cat([model_input, output[:, -1:, :]], dim=1)

        return model_input[:, 1:], cond_dict

    # def on_validation_epoch_start(self) -> None:
    #     # Use text as condition during validation
    #     for key in self.cond_stage_model_metadata.keys():
    #         metadata = self.cond_stage_model_metadata[key]
    #         model_idx, cond_stage_key, conditioning_key = metadata["model_idx"], metadata["cond_stage_key"], metadata["conditioning_key"]

    #         # If we use CLAP as condition, we might use audio for training, but we also must use text for evaluation
    #         # if(isinstance(self.cond_stage_models[model_idx], CLAPAudioEmbeddingClassifierFreev2)):
    #         #     self.cond_stage_model_metadata[key]["cond_stage_key_orig"] = self.cond_stage_model_metadata[key]["cond_stage_key"]
    #         #     self.cond_stage_model_metadata[key]["embed_mode_orig"] = self.cond_stage_models[model_idx].embed_mode
    #         #     print("Change the model original cond_keyand embed_mode %s, %s to text during evaluation" % (self.cond_stage_model_metadata[key]["cond_stage_key_orig"], self.cond_stage_model_metadata[key]["embed_mode_orig"]))
    #         #     self.cond_stage_model_metadata[key]["cond_stage_key"] = "text"
    #         #     self.cond_stage_models[model_idx].embed_mode = "text"

    #     return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        cond_dict = self.get_input(batch)
        # cond_dict['film_clap_cond1']: [2,1,512]
        # cond_dict['crossattn_audiomae_pooled']: [2, 128, 768]

        input_embeds, target_embeds = (
            cond_dict["film_clap_cond1"],
            cond_dict["crossattn_audiomae_pooled"][0],
        )

        # Some times if the pooling factor is random, the length of crossattn_audiomae_pooled is not necessary 32, so need to calculate separately
        if "crossattn_audiomae_pooled_44" in cond_dict.keys():
            target_embeds = cond_dict["crossattn_audiomae_pooled_44"][0]

        if self.use_audiomae_linear:
            input_embeds = torch.cat(
                [self.linear_clap(input_embeds), self.linear_audiomae(target_embeds)],
                dim=1,
            )
        else:
            input_embeds = torch.cat(
                [self.linear_clap(input_embeds), target_embeds], dim=1
            )

        output_embeds = self.model(inputs_embeds=input_embeds)["last_hidden_state"]

        target = target_embeds
        output = output_embeds[:, :-1]

        loss = self.loss_fn(output, target)

        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            sync_dist=True,
            on_epoch=True,
        )

        generation_output, _ = self.generate(batch)
        ar_gen_loss = self.loss_fn(generation_output, target)

        self.log(
            "val/ar_gen_loss",
            ar_gen_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            sync_dist=True,
            on_epoch=True,
        )

        return {"loss": loss, "ar_gen_loss": ar_gen_loss}

    def get_input_item(self, batch, k):
        fname, text, label_indices, waveform, stft, fbank = (
            batch["fname"],
            batch["text"],
            batch["label_vector"],
            batch["waveform"],
            batch["stft"],
            batch["log_mel_spec"],
        )
        ret = {}

        ret["fbank"] = (
            fbank.unsqueeze(1).to(memory_format=torch.contiguous_format).float()
        )
        ret["stft"] = stft.to(memory_format=torch.contiguous_format).float()
        # ret["clip_label"] = clip_label.to(memory_format=torch.contiguous_format).float()
        ret["waveform"] = waveform.to(memory_format=torch.contiguous_format).float()
        ret["text"] = list(text)
        ret["fname"] = fname

        for key in batch.keys():
            if key not in ret.keys():
                ret[key] = batch[key]

        return ret[k]

    def get_input(self, batch):
        cond_dict = {}
        if len(self.cond_stage_model_metadata.keys()) > 0:
            unconditional_cfg = False

            for cond_model_key in self.cond_stage_model_metadata.keys():
                cond_stage_key = self.cond_stage_model_metadata[cond_model_key][
                    "cond_stage_key"
                ]

                # if(not self.training):
                #     if(isinstance(self.cond_stage_models[self.cond_stage_model_metadata[cond_model_key]["model_idx"]], CLAPAudioEmbeddingClassifierFreev2)):
                #         assert cond_stage_key == "text" # CLAP model should use text for evaluation

                # The original data for conditioning
                xc = self.get_input_item(batch, cond_stage_key)
                if type(xc) == torch.Tensor:
                    xc = xc.to(self.device)

                c = self.get_learned_conditioning(
                    xc, key=cond_model_key, unconditional_cfg=unconditional_cfg
                )
                cond_dict[cond_model_key] = c

        return cond_dict

    def instantiate_cond_stage(self, config):
        self.cond_stage_model_metadata = {}

        for i, cond_model_key in enumerate(config.keys()):
            model = instantiate_from_config(config[cond_model_key])
            self.cond_stage_models.append(model)
            self.cond_stage_model_metadata[cond_model_key] = {
                "model_idx": i,
                "cond_stage_key": config[cond_model_key]["cond_stage_key"],
                "conditioning_key": config[cond_model_key]["conditioning_key"],
            }

    def get_learned_conditioning(self, c, key, unconditional_cfg):
        assert key in self.cond_stage_model_metadata.keys()

        # Classifier-free guidance
        if not unconditional_cfg:
            c = self.cond_stage_models[
                self.cond_stage_model_metadata[key]["model_idx"]
            ](c)
        else:
            if isinstance(c, torch.Tensor):
                batchsize = c.size(0)
            elif isinstance(c, list):
                batchsize = len(c)
            else:
                raise NotImplementedError()
            c = self.cond_stage_models[
                self.cond_stage_model_metadata[key]["model_idx"]
            ].get_unconditional_condition(batchsize)

        return c
