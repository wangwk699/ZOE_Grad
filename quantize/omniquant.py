import torch
import torch.nn as nn
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc

from models.int_qwen3_layer import QuantQwen3DecoderLayer
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_llama3_layer import QuantLlama3DecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer

from quantize.int_linear import QuantLinear
from quantize.utils import (
    let_parameters,
    lwc_parameters,
    get_omni_parameters,
    omni_state_dict,
    register_scales_and_zeros,
    smooth_and_quant_temporary,
    smooth_and_quant_inplace,
    clear_temp_variable,
    set_quant_state,
)

# Qwen3 needs per-layer mask mapping
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask



def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")

    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False

    is_llama = False
    is_llama3 = False
    is_qwen = False
    is_qwen3 = False

    net_name = args.net.lower()

    if "llama" in net_name:
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

        # llama3 uses the new v4.57.6 style decoder interface:
        # shared causal_mask + shared position_embeddings
        if "llama-3" in net_name or "llama3" in net_name:
            is_llama3 = True
            DecoderLayer = QuantLlama3DecoderLayer
        else:
            DecoderLayer = QuantLlamaDecoderLayer

        pairs = {
            "q_proj": "qkv",
            "o_proj": "out",
            "up_proj": "fc1",
        }
        layer_name_prefix = "model.layers"

    elif "qwen" in net_name:
        is_qwen = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

        if "qwen3" not in net_name:
            raise ValueError("Only Qwen3-8B is supported")
        is_qwen3 = True
        DecoderLayer = QuantQwen3DecoderLayer

        pairs = {
            "q_proj": "qkv",
            "o_proj": "out",
            "up_proj": "fc1",
        }
        layer_name_prefix = "model.layers"

    elif "opt" in net_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "out_proj": "out",
            "fc1": "fc1",
        }
        layer_name_prefix = "model.decoder.layers"

    else:
        raise ValueError("Only OPT, Llama-2, Llama-3 and Qwen3 are supported")

    layers[0] = layers[0].to(dev)

    if args.deactive_amp and args.epochs > 0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = lambda: torch.amp.autocast("cuda")

    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )

    cache = {"i": 0}

    # Attention state for Llama and Qwen3
    attention_mask = None
    attention_mask_batch = None
    position_ids = None
    position_embeddings = None

    # Qwen3 only
    attention_mask_mapping = None

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            self.is_llama3 = False
            self.is_qwen3 = False

            # Qwen3Model.forward reads decoder_layer.attention_type before calling the layer
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1

            if self.is_llama3:
                if "attention_mask" in kwargs:
                    cache["attention_mask"] = kwargs["attention_mask"]
                if "position_embeddings" in kwargs:
                    cache["position_embeddings"] = kwargs["position_embeddings"]
                if "position_ids" in kwargs:
                    cache["position_ids"] = kwargs["position_ids"]

            elif self.is_llama:
                if "attention_mask" in kwargs:
                    cache["attention_mask"] = kwargs["attention_mask"]
                if "position_ids" in kwargs:
                    cache["position_ids"] = kwargs["position_ids"]

            elif self.is_qwen3:
                # Qwen3 first layer only sees the selected mask for its own attention_type,
                # so the full mapping is built outside and position_embeddings are cached here.
                if "position_embeddings" in kwargs:
                    cache["position_embeddings"] = kwargs["position_embeddings"]
                if "position_ids" in kwargs:
                    cache["position_ids"] = kwargs["position_ids"]

            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama and not is_llama3
    layers[0].is_llama3 = is_llama3
    layers[0].is_qwen3 = is_qwen3

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break

            input_ids = batch[0].to(dev)

            try:
                if is_qwen3:
                    # Qwen3: build the full mask mapping once, because each layer may use a different mask
                    if attention_mask_mapping is None:
                        inputs_embeds = model.model.embed_tokens(input_ids)

                        past_key_values = None
                        past_seen_tokens = 0
                        cache_position = torch.arange(
                            past_seen_tokens,
                            past_seen_tokens + inputs_embeds.shape[1],
                            device=inputs_embeds.device,
                        )
                        position_ids = cache_position.unsqueeze(0)

                        mask_kwargs = {
                            "config": model.model.config,
                            "input_embeds": inputs_embeds,
                            "attention_mask": None,
                            "cache_position": cache_position,
                            "past_key_values": past_key_values,
                            "position_ids": position_ids,
                        }

                        attention_mask_mapping = {
                            "full_attention": create_causal_mask(**mask_kwargs),
                        }

                        if "sliding_attention" in model.model.config.layer_types:
                            attention_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

                        position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_mapping,
                        position_ids=position_ids,
                    )

                else:
                    # let the HF model build its own mask and pass it into layer 0, then Catcher caches it
                    model(input_ids)

            except ValueError:
                pass

    # restore the first layer
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    if "llama" in net_name or "qwen" in net_name:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in net_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    else:
        raise ValueError("Only OPT, Llama-2, Llama-3 and Qwen3 are supported")

    torch.cuda.empty_cache()

    quant_inps = inps
    fp_inps = copy.deepcopy(inps)
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None

    loss_func = torch.nn.MSELoss()

    if is_llama3:
        attention_mask = cache.get("attention_mask", None)
        position_embeddings = cache.get("position_embeddings", None)
        position_ids = cache.get("position_ids", None)

        if attention_mask is not None:
            attention_mask_batch = (
                attention_mask.repeat(args.batch_size, 1, 1, 1)
                if args.deactive_amp
                else attention_mask.repeat(args.batch_size, 1, 1, 1).float()
            )
        else:
            logger.info(
                "No attention mask caught from the first layer."
                " Seems that model's attention works without a mask."
            )
            attention_mask_batch = None

    elif is_llama:
        attention_mask = cache.get("attention_mask", None)
        position_ids = cache.get("position_ids", None)

        if attention_mask is not None:
            attention_mask_batch = (
                attention_mask.repeat(args.batch_size, 1, 1, 1)
                if args.deactive_amp
                else attention_mask.repeat(args.batch_size, 1, 1, 1).float()
            )
        else:
            logger.info(
                "No attention mask caught from the first layer."
                " Seems that model's attention works without a mask."
            )
            attention_mask_batch = None

    elif is_qwen3:
        position_embeddings = cache.get("position_embeddings", position_embeddings)
        position_ids = cache.get("position_ids", position_ids)

    else:
        position_ids = None

    if args.resume:
        omni_parameters = torch.load(args.resume, map_location="cpu", weights_only=False)
    else:
        omni_parameters = {}

    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)

        qlayer = DecoderLayer(lm.model.config, layer, args)

        qlayer = qlayer.to(dev)

        # current layer attention mask
        if is_qwen3:
            layer_attention_mask = attention_mask_mapping[qlayer.attention_type]
        else:
            layer_attention_mask = attention_mask
            layer_attention_mask_batch = attention_mask_batch

        # obtain full-precision outputs
        set_quant_state(qlayer, weight_quant=False, act_quant=False)

        if args.epochs > 0:
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    for j in range(args.nsamples):
                        if is_qwen3 or is_llama3:
                            fp_inps[j] = qlayer(
                                fp_inps[j].unsqueeze(0),
                                attention_mask=layer_attention_mask,
                                position_embeddings=position_embeddings,
                            )
                            if args.aug_loss:
                                fp_inps_2[j] = qlayer(
                                    quant_inps[j].unsqueeze(0),
                                    attention_mask=layer_attention_mask,
                                    position_embeddings=position_embeddings,
                                )
                        else:
                            fp_inps[j] = qlayer(
                                fp_inps[j].unsqueeze(0),
                                attention_mask=layer_attention_mask,
                                position_ids=position_ids,
                            )[0]
                            if args.aug_loss:
                                fp_inps_2[j] = qlayer(
                                    quant_inps[j].unsqueeze(0),
                                    attention_mask=layer_attention_mask,
                                    position_ids=position_ids,
                                )[0]

        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)
        qlayer.let = args.let

        use_shift = True
        if (is_llama or is_qwen) or args.abits == 16:
            use_shift = False

        if args.let:
            qlayer.register_parameter(
                "qkt_smooth_scale",
                torch.nn.Parameter(
                    torch.ones(
                        layer.self_attn.q_proj.out_features,
                        device=dev,
                        dtype=dtype,
                    )
                ),
            )

            for name, module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha) / weight.pow(1 - args.alpha)).clamp(min=1e-5)

                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)

                            qlayer.register_parameter(
                                f"{pairs[key]}_smooth_shift",
                                torch.nn.Parameter(shift),
                            )
                            qlayer.register_parameter(
                                f"{pairs[key]}_smooth_scale",
                                torch.nn.Parameter(scale),
                            )

        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()

            optimizer = torch.optim.AdamW(
                [
                    {"params": let_parameters(qlayer, use_shift), "lr": args.let_lr},
                    {"params": lwc_parameters(qlayer), "lr": args.lwc_lr},
                ],
                weight_decay=args.wd,
            )
            loss_scaler = utils.NativeScalerWithGradNormCount()

            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []

                for j in range(args.nsamples // args.batch_size):
                    index = j * args.batch_size

                    with traincast():
                        if is_qwen3:
                            # qwen3 keeps a single per-layer mask and relies on broadcast
                            smooth_and_quant_temporary(qlayer, args, True)
                            quant_out = qlayer(
                                quant_inps[index:index + args.batch_size],
                                attention_mask=layer_attention_mask,
                                position_embeddings=position_embeddings,
                            )

                        elif is_llama3:
                            # llama3 uses the same causal mask for all layers
                            smooth_and_quant_temporary(qlayer, args, True)
                            quant_out = qlayer(
                                quant_inps[index:index + args.batch_size],
                                attention_mask=layer_attention_mask_batch,
                                position_embeddings=position_embeddings,
                            )

                        else:
                            is_llama_qwen = is_llama or is_qwen
                            smooth_and_quant_temporary(qlayer, args, is_llama_qwen)
                            quant_out = qlayer(
                                quant_inps[index:index + args.batch_size],
                                attention_mask=layer_attention_mask_batch,
                                position_ids=position_ids,
                            )[0]

                        loss = loss_func(fp_inps[index:index + args.batch_size], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index + args.batch_size], quant_out)

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()

                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(
                        loss,
                        optimizer,
                        parameters=get_omni_parameters(qlayer, use_shift),
                    ).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(
                    f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} "
                    f"max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2}"
                )

            clear_temp_variable(qlayer)
            del optimizer

        qlayer.half()

        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)

        if args.epochs > 0:
            with torch.no_grad():
                with traincast():
                    for j in range(args.nsamples):
                        if is_qwen3 or is_llama3:
                            quant_inps[j] = qlayer(
                                quant_inps[j].unsqueeze(0),
                                attention_mask=layer_attention_mask,
                                position_embeddings=position_embeddings,
                            )
                        else:
                            quant_inps[j] = qlayer(
                                quant_inps[j].unsqueeze(0),
                                attention_mask=layer_attention_mask,
                                position_ids=position_ids,
                            )[0]

            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, "omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")

        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()

    model.config.use_cache = use_cache
    return model