import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from audioldm_train.utilities.model_util import (
    exists,
    default,
    mean_flat,
    count_params,
    instantiate_from_config,
)
from torch.optim import *

from transformers import GPT2Config, GPT2Model, GPTJConfig, GPTJModel
import torch.optim.lr_scheduler as lr_scheduler


class Sequence2AudioMAE(pl.LightningModule):
    def __init__(
        self,
        base_learning_rate,
        sequence_gen_length,
        sequence_input_key,
        sequence_input_embed_dim,
        cond_stage_config,
        optimizer_type="AdamW",
        use_warmup=True,
        use_ar_gen_loss=False,
        use_audiomae_linear=False,
        target_tokens_mask_ratio=0.0,
        random_mask_ratio=False,
        **kwargs
    ):

        super().__init__()
        assert use_audiomae_linear == False
        self.random_mask_ratio = random_mask_ratio
        self.learning_rate = base_learning_rate
        self.cond_stage_config = cond_stage_config
        self.use_audiomae_linear = use_audiomae_linear
        self.optimizer_type = optimizer_type
        self.use_warmup = use_warmup
        self.use_ar_gen_loss = use_ar_gen_loss
        # Even though the LDM can be conditioned on mutliple pooling rate
        # Our model always predict the higest pooling rate

        self.mae_token_num = sequence_gen_length
        self.sequence_input_key = sequence_input_key
        self.sequence_input_embed_dim = sequence_input_embed_dim
        self.target_tokens_mask_ratio = target_tokens_mask_ratio

        self.start_of_sequence_tokens = nn.Embedding(32, 768)
        self.end_of_sequence_tokens = nn.Embedding(32, 768)

        self.input_sequence_embed_linear = nn.ModuleList([])
        self.initial_learning_rate = None

        for dim in self.sequence_input_embed_dim:
            self.input_sequence_embed_linear.append(nn.Linear(dim, 768))

        self.cond_stage_models = nn.ModuleList([])
        self.instantiate_cond_stage(cond_stage_config)
        self.initialize_param_check_toolkit()

        self.private_training_step = 0

        # configuration = GPT2Config(n_layer=1) # TODO
        # self.model=GPT2Model(configuration)
        ###################
        # self.model=nn.Linear(768,768, bias=False) # TODO change the model
        # with torch.no_grad():
        #     self.model.weight.copy_(torch.eye(768))
        ###################
        self.model = GPT2Model.from_pretrained("gpt2")
        ###################
        # self.model = nn.LSTM(input_size=768, hidden_size=768, num_layers=1,bias=False) # TODO

        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.L1Loss()

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
        # params = list(self.model.parameters()) + list(self.input_sequence_embed_linear.parameters())
        params = list(self.parameters())

        # opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.98), eps=1e-9)
        opt = eval(self.optimizer_type)(params, lr=lr)
        scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.8)
        return [opt], [scheduler]

    def add_sos_eos_tokens(self, _id, sequence, attn_mask):
        batchsize = sequence.size(0)

        new_attn_mask_step = torch.ones((batchsize, 1)).to(sequence.device)
        key_id = torch.tensor([_id]).to(sequence.device)

        # Add two more steps to attn mask
        new_attn_mask = torch.cat(
            [new_attn_mask_step, attn_mask, new_attn_mask_step], dim=1
        )

        # Add two more tokens in the sequence
        sos_token = self.start_of_sequence_tokens(key_id).expand(batchsize, 1, -1)
        eos_token = self.end_of_sequence_tokens(key_id).expand(batchsize, 1, -1)
        new_sequence = torch.cat([sos_token, sequence, eos_token], dim=1)
        return new_sequence, new_attn_mask

    def truncate_sequence_and_mask(self, sequence, mask, max_len=512):
        if sequence.size(1) > max_len:
            print(
                "The input sequence length to GPT-2 model is too long:",
                sequence.size(1),
            )
            return sequence[:, :max_len], mask[:, :max_len]
        else:
            return sequence, mask

    def get_input_sequence_and_mask(self, cond_dict):
        input_embeds = None
        input_embeds_attn_mask = None
        for _id, sequence_key in enumerate(self.sequence_input_key):
            assert sequence_key in cond_dict.keys(), (
                "Invalid sequence key %s" % sequence_key
            )
            cond_embed = cond_dict[sequence_key]
            if isinstance(cond_embed, list):
                assert (
                    len(cond_embed) == 2
                ), "The crossattn returned list should have length 2, including embed and attn_mask"
                item_input_embeds, item_attn_mask = cond_embed

                item_input_embeds = self.input_sequence_embed_linear[_id](
                    item_input_embeds
                )

                item_input_embeds, item_attn_mask = self.add_sos_eos_tokens(
                    _id, item_input_embeds, item_attn_mask
                )

                if input_embeds is None and input_embeds_attn_mask is None:
                    input_embeds, input_embeds_attn_mask = (
                        item_input_embeds,
                        item_attn_mask,
                    )
                else:
                    input_embeds = torch.cat(
                        [input_embeds, item_input_embeds], dim=1
                    )  # The 1-st dimension is time steps
                    input_embeds_attn_mask = torch.cat(
                        [input_embeds_attn_mask, item_attn_mask], dim=1
                    )  # The 1-st dimension is time steps
            else:
                assert isinstance(cond_embed, torch.Tensor)
                cond_embed = self.input_sequence_embed_linear[_id](cond_embed)
                attn_mask = torch.ones((cond_embed.size(0), cond_embed.size(1))).to(
                    cond_embed.device
                )

                item_input_embeds, item_attn_mask = self.add_sos_eos_tokens(
                    _id, cond_embed, attn_mask
                )

                if input_embeds is None and input_embeds_attn_mask is None:
                    input_embeds, input_embeds_attn_mask = (
                        item_input_embeds,
                        item_attn_mask,
                    )
                else:
                    input_embeds, input_embeds_attn_mask = torch.cat(
                        [input_embeds, item_input_embeds], dim=1
                    ), torch.cat([input_embeds_attn_mask, item_attn_mask], dim=1)

        assert input_embeds is not None and input_embeds_attn_mask is not None

        input_embeds, input_embeds_attn_mask = self.truncate_sequence_and_mask(
            input_embeds, input_embeds_attn_mask, int(1024 - self.mae_token_num)
        )
        cond_sequence_end_time_idx = input_embeds.size(
            1
        )  # The index that we start to collect the output embeds

        return input_embeds, input_embeds_attn_mask, cond_sequence_end_time_idx

    def warmup_step(self):
        if self.initial_learning_rate is None:
            self.initial_learning_rate = float(self.learning_rate)

        # Only the first parameter group
        if self.global_step <= 1000:
            if self.global_step == 0:
                print(
                    "Warming up learning rate start with %s"
                    % self.initial_learning_rate
                )
            self.trainer.optimizers[0].param_groups[0]["lr"] = (
                self.global_step / 1000
            ) * self.initial_learning_rate
        else:
            # TODO set learning rate here
            self.trainer.optimizers[0].param_groups[0][
                "lr"
            ] = self.initial_learning_rate

    def mask_target_sequence(self, target_embeds, target_embeds_attn_mask):
        time_seq_mask = None
        if self.target_tokens_mask_ratio > 1e-4:
            batchsize, time_seq_len, embed_dim = target_embeds.size()
            _, time_seq_len = target_embeds_attn_mask.size()
            # Generate random mask
            if self.random_mask_ratio:
                mask_ratio = torch.rand(1).item() * self.target_tokens_mask_ratio
            else:
                mask_ratio = self.target_tokens_mask_ratio

            time_seq_mask = (torch.rand((batchsize, time_seq_len)) > mask_ratio).to(
                target_embeds.device
            )
            # Mask the target embedding
            target_embeds = target_embeds * time_seq_mask.unsqueeze(-1)
            target_embeds_attn_mask = target_embeds_attn_mask * time_seq_mask
        return target_embeds, target_embeds_attn_mask, time_seq_mask

    def training_step(self, batch, batch_idx=None, cond_dict=None, return_output=False):
        # cond_dict['film_clap_cond1']: [2,1,512]
        # cond_dict['crossattn_audiomae_pooled']: [2, 128, 768]

        if self.use_warmup:
            self.warmup_step()

        if cond_dict is None:
            cond_dict = self.get_input(batch)

        # param_list = list(self.model.parameters())
        target_embeds, target_embeds_attn_mask = (
            cond_dict["crossattn_audiomae_pooled"][0],
            cond_dict["crossattn_audiomae_pooled"][1],
        )

        (
            input_embeds,
            input_embeds_attn_mask,
            cond_sequence_end_time_idx,
        ) = self.get_input_sequence_and_mask(cond_dict)

        # Some times if the pooling factor is random, the length of crossattn_audiomae_pooled is not necessary 32, so need to calculate separately
        if "crossattn_audiomae_pooled_44" in cond_dict.keys():
            target_embeds = cond_dict["crossattn_audiomae_pooled_44"][0]

        # target_embeds, target_embeds_attn_mask, time_seq_mask = self.mask_target_sequence(target_embeds, target_embeds_attn_mask)

        final_input_embeds = torch.cat([input_embeds, target_embeds], dim=1)
        final_input_embeds_attn_mask = torch.cat(
            [input_embeds_attn_mask, target_embeds_attn_mask], dim=1
        )

        ########################### GPT-2
        output_embeds = self.model(
            inputs_embeds=final_input_embeds,
            attention_mask=final_input_embeds_attn_mask,
        )["last_hidden_state"]
        ########################### DNN
        # output_embeds = self.model(final_input_embeds)
        ########################### LSTM
        # output_embeds,_ = self.model(final_input_embeds)

        target = target_embeds
        output = output_embeds[:, cond_sequence_end_time_idx - 1 : -1]

        # output = output_embeds[:, cond_sequence_end_time_idx: ] # TODO bug here intentionally

        assert target.size(1) == self.mae_token_num

        # if(batch_idx % 1000 == 0):
        #     print(output[0], target[0])
        loss = self.loss_fn(output, target)

        if self.use_ar_gen_loss:
            ar_gen_loss = self.calculate_ahead_k_step_loss(batch, batch_idx, cond_dict)
        else:
            ar_gen_loss = loss

        if self.private_training_step % 500 == 0:
            print(
                "AudioMAE prediction module:", "loss", loss, "ar_gen_loss", ar_gen_loss
            )

        try:
            learning_rate = self.trainer.optimizers[0].param_groups[0]["lr"]

            self.log(
                "train/lr_audiomae_pred",
                learning_rate,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        except:
            pass

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
            "train/loss_ar_gen_loss",
            ar_gen_loss,
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
        self.private_training_step += 1
        if return_output:
            return loss + ar_gen_loss, output
        else:
            return loss + ar_gen_loss

    def calculate_ahead_k_step_loss(self, batch, batch_idx=None, cond_dict=None):
        if cond_dict is None:
            cond_dict = self.get_input(batch)

        target_embeds, target_embeds_attn_mask = (
            cond_dict["crossattn_audiomae_pooled"][0],
            cond_dict["crossattn_audiomae_pooled"][1],
        )

        assert (
            torch.sum(target_embeds_attn_mask < 0.1) < 1
        ), "This function only works for AudioMAE prediction, which should have all one atten_mask"

        (
            input_embeds,
            input_embeds_attn_mask,
            cond_sequence_end_time_idx,
        ) = self.get_input_sequence_and_mask(cond_dict)

        target_total_time_steps = target_embeds.size(1)

        steps = min(round(torch.rand(1).item() * 8), target_total_time_steps)

        if steps < 2:
            steps = 2

        start_idx = max(
            0, round(torch.rand(1).item() * (target_total_time_steps - steps)) - 1
        )

        model_input = input_embeds
        model_input_mask = input_embeds_attn_mask
        target_embeds_ar_gen = target_embeds[:, start_idx : start_idx + steps, :]
        generation = []

        if start_idx > 0:
            model_input = torch.cat(
                [input_embeds, target_embeds[:, :start_idx, :]], dim=1
            )
            attention_mask_known_steps = torch.ones(
                (model_input_mask.size(0), start_idx)
            ).to(model_input.device)
            model_input_mask = torch.cat(
                [input_embeds_attn_mask, attention_mask_known_steps], dim=1
            )

        for _ in range(steps):
            output = self.model(
                inputs_embeds=model_input, attention_mask=model_input_mask
            )["last_hidden_state"]
            # Update the model input
            generation.append(output[:, -1:, :])
            model_input = torch.cat([model_input, output[:, -1:, :]], dim=1)
            # Update the attention mask
            attention_mask_new_step = torch.ones((model_input_mask.size(0), 1)).to(
                model_input.device
            )
            model_input_mask = torch.cat(
                [model_input_mask, attention_mask_new_step], dim=1
            )

        generation = torch.cat(generation, dim=1)

        return self.loss_fn(generation, target_embeds_ar_gen)

    def generate_partial(self, batch, cond_dict=None, no_grad=False):
        if cond_dict is None:
            cond_dict = self.get_input(batch)

        print("Generate partially prompted audio with in-context learning")
        # self.model.train()
        # assert self.model.training==True

        target_embeds, target_embeds_attn_mask = (
            cond_dict["crossattn_audiomae_pooled"][0],
            cond_dict["crossattn_audiomae_pooled"][1],
        )

        target_time_steps = target_embeds.size(1)

        (
            input_embeds,
            input_embeds_attn_mask,
            cond_sequence_end_time_idx,
        ) = self.get_input_sequence_and_mask(cond_dict)

        model_input = torch.cat(
            [input_embeds, target_embeds[:, : target_time_steps // 4, :]], dim=1
        )
        model_input_mask = torch.cat(
            [
                input_embeds_attn_mask,
                target_embeds_attn_mask[:, : target_time_steps // 4],
            ],
            dim=1,
        )

        steps = self.mae_token_num

        for _ in range(3 * steps // 4):
            output = self.model(
                inputs_embeds=model_input, attention_mask=model_input_mask
            )["last_hidden_state"]
            # Update the model input
            model_input = torch.cat([model_input, output[:, -1:, :]], dim=1)
            # Update the attention mask
            attention_mask_new_step = torch.ones((model_input_mask.size(0), 1)).to(
                model_input.device
            )
            model_input_mask = torch.cat(
                [model_input_mask, attention_mask_new_step], dim=1
            )

        output = model_input[:, cond_sequence_end_time_idx:]

        return output, cond_dict

    def generate(self, batch, cond_dict=None, no_grad=False):
        if cond_dict is None:
            cond_dict = self.get_input(batch)

        # self.model.train()
        # print("!!!!!!!!!!!!!train")

        (
            input_embeds,
            input_embeds_attn_mask,
            cond_sequence_end_time_idx,
        ) = self.get_input_sequence_and_mask(cond_dict)
        model_input = input_embeds
        model_input_mask = input_embeds_attn_mask

        steps = self.mae_token_num

        for _ in range(steps):
            output = self.model(
                inputs_embeds=model_input, attention_mask=model_input_mask
            )["last_hidden_state"]
            # Update the model input
            model_input = torch.cat([model_input, output[:, -1:, :]], dim=1)
            # Update the attention mask
            attention_mask_new_step = torch.ones((model_input_mask.size(0), 1)).to(
                model_input.device
            )
            model_input_mask = torch.cat(
                [model_input_mask, attention_mask_new_step], dim=1
            )

        return model_input[:, cond_sequence_end_time_idx:], cond_dict

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

        target_embeds, target_embeds_attn_mask = (
            cond_dict["crossattn_audiomae_pooled"][0],
            cond_dict["crossattn_audiomae_pooled"][1],
        )

        (
            input_embeds,
            input_embeds_attn_mask,
            cond_sequence_end_time_idx,
        ) = self.get_input_sequence_and_mask(cond_dict)

        # Some times if the pooling factor is random, the length of crossattn_audiomae_pooled is not necessary 32, so need to calculate separately
        if "crossattn_audiomae_pooled_44" in cond_dict.keys():
            target_embeds = cond_dict["crossattn_audiomae_pooled_44"][0]

        final_input_embeds = torch.cat([input_embeds, target_embeds], dim=1)
        final_input_embeds_attn_mask = torch.cat(
            [input_embeds_attn_mask, target_embeds_attn_mask], dim=1
        )

        output_embeds = self.model(
            inputs_embeds=final_input_embeds,
            attention_mask=final_input_embeds_attn_mask,
        )["last_hidden_state"]

        target = target_embeds
        output = output_embeds[:, cond_sequence_end_time_idx - 1 : -1]

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

    def initialize_param_check_toolkit(self):
        self.tracked_steps = 0
        self.param_dict = {}

    def statistic_require_grad_tensor_number(self, module, name=None):
        requires_grad_num = 0
        total_num = 0
        require_grad_tensor = None
        for p in module.parameters():
            if p.requires_grad:
                requires_grad_num += 1
                if require_grad_tensor is None:
                    require_grad_tensor = p
            total_num += 1
        print(
            "Module: [%s] have %s trainable parameters out of %s total parameters (%.2f)"
            % (name, requires_grad_num, total_num, requires_grad_num / total_num)
        )
        return require_grad_tensor

    def check_module_param_update(self):

        if self.tracked_steps == 0:
            print("Sequence2AudioMAE")
            for name, module in self.named_children():
                try:
                    require_grad_tensor = self.statistic_require_grad_tensor_number(
                        module, name=name
                    )
                    if require_grad_tensor is not None:
                        self.param_dict[name] = require_grad_tensor.clone()
                    else:
                        print("==> %s does not requires grad" % name)
                except Exception as e:
                    print("%s does not have trainable parameters: %s" % (name, e))
                    continue

        if self.tracked_steps % 5000 == 0:
            print("Sequence2AudioMAE")
            for name, module in self.named_children():
                try:
                    require_grad_tensor = self.statistic_require_grad_tensor_number(
                        module, name=name
                    )

                    if require_grad_tensor is not None:
                        print(
                            "===> Param diff %s: %s; Size: %s"
                            % (
                                name,
                                torch.sum(
                                    torch.abs(
                                        self.param_dict[name] - require_grad_tensor
                                    )
                                ),
                                require_grad_tensor.size(),
                            )
                        )
                    else:
                        print("%s does not requires grad" % name)
                except Exception as e:
                    print("%s does not have trainable parameters: %s" % (name, e))
                    continue

        self.tracked_steps += 1
