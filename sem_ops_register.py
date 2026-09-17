from typing import List
from pyspark.sql import DataFrame, Column
from pyspark.sql.functions import col as pyspark_col
import re
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import time
from PIL import Image
import io
import base64
import time
from typing import List, Dict, Tuple, Optional, Any, Sequence
import numpy as np
from PIL import Image
import io
import time

DEFAULT_SYSTEM_PROMPT = """
You are a helpful data analyst. You will receive data containing various fields and their corresponding values, representing different attributes. Use these fields to provide an answer to the user query. The user query will indicate which fields to use for your response. Your response should contain only the answer and no additional formatting.
"""

def get_visual_similarity(image_features):
    """
    Compute the cosine similarity matrix among image features.
    """
    image_features = image_features.float()  # (B, N, D)
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)  # (B, N, D)
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))  # (B, N, N)
    return similarity


def get_relevance_score(image_embeds, text_embeds):
    """
    Compute the relevance score between image and text embeddings.
    """
    image_embeds = image_embeds.float()
    text_embeds = text_embeds.float()
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)  # (B, N, C)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # (M, C)

    relevance = torch.matmul(image_embeds, text_embeds.t())  # (B, N, M)
    relevance = (-relevance).mean(dim=-1)  # (B, N)
    relevance = (relevance - relevance.min(dim=1, keepdim=True)[0]) / (
        relevance.max(dim=1, keepdim=True)[0] - relevance.min(dim=1, keepdim=True)[0] + 1e-6
    )
    return relevance


def build_conditional_kernel_matrix(relevance, similarity, theta=0.5):
    """
    Build the conditional DPP kernel matrix based on relevance and visual similarity.
    """
    if theta != 1:
        alpha = theta / (2 * (1 - theta))
        relevance = torch.exp(alpha * relevance)  # (B, N)

    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)  # (B, N, N)
    return kernel


def conditional_dpp_map(kernel, num_keep_tokens):
    """
    Perform conditional DPP MAP inference to select a subset of tokens.
    """
    device = kernel.device

    # kernel diagonal (di2s[b, i] = kernel[b, i, i] = relevance[b, i] ** 2 * (L[i,i]=1))
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()

    # orthogonal directions corresponding to selected tokens (L~=CC^T)
    B, N = di2s.shape
    cis = torch.zeros((num_keep_tokens, B, N), device=device)  # (num_keep_tokens, B, N)

    keep_indices = torch.empty((num_keep_tokens, B), dtype=torch.long, device=device)
    batch_idx = torch.arange(B)
    for i in range(num_keep_tokens):
        j = torch.argmax(di2s, dim=-1)  # Select the index with highest remaining score
        keep_indices[i] = j

        # compute the orthogonalized row vector for token j
        if i == 0:
            eis = kernel[batch_idx, j] / torch.sqrt(kernel[batch_idx, j, j].unsqueeze(1) + 1e-5)  # (B, N)
        else:
            proj = torch.einsum("tb,tbn->bn", cis[:i, batch_idx, j], cis[:i])  # (B, N)
            eis = (kernel[batch_idx, j] - proj) / torch.sqrt(di2s[batch_idx, j].unsqueeze(-1) + 1e-5)

        cis[i, :, :] = eis
        di2s -= eis**2
        di2s[batch_idx, j] = -float("inf")

    keep_indices = torch.sort(keep_indices.t()).values
    return keep_indices


def get_model_kwargs(model, inputs):
    """
    Get the model keyword arguments from the model and inputs.
    """
    kwargs = {}
    if hasattr(model.config, "vision_feature_select_strategy"):
        kwargs["vision_feature_select_strategy"] = model.config.vision_feature_select_strategy
    if hasattr(model.config, "vision_feature_layer"):
        kwargs["vision_feature_layer"] = model.config.vision_feature_layer
    if hasattr(inputs, "image_sizes"):
        kwargs["image_sizes"] = inputs.image_sizes
    if hasattr(inputs, "image_grid_thw"):
        kwargs["image_grid_thw"] = inputs.image_grid_thw
    return kwargs


def get_image_features(model, inputs, **kwargs):
    """
    Extract image features from the model.
    """
    pixel_values = inputs.pixel_values
    image_num_patches = None
    if "LlavaNextForConditionalGeneration" in model.config.architectures and pixel_values.dim() == 5:
        from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches

        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=model.config.image_grid_pinpoints,
                patch_size=model.config.vision_config.image_size,
            )
            for imsize in inputs.image_sizes
        ]
        # stacked if input is (batch_size, num_patches, num_channels, height, width)
        _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
        pixel_values = torch.cat(_pixel_values_list, dim=0)

    if hasattr(model, "vision_tower"):  # llava or internvl
        vision_feature_layer = kwargs.get("vision_feature_layer", -1)
        image_features = model.vision_tower(pixel_values=pixel_values, output_hidden_states=True)  # .last_hidden_state
        image_features = image_features.hidden_states[vision_feature_layer]

        if kwargs.get("vision_feature_select_strategy") == "default":
            image_features = image_features[:, 1:]  # remove CLS token
        if image_num_patches is None:  # LlavaForConditionalGeneration
            image_num_patches = image_features.shape[0]
    elif hasattr(model, "visual") or hasattr(model, "vision_tower"):  # qwen
        image_features = model.visual.patch_embed(pixel_values)
        image_num_patches = (kwargs["image_grid_thw"].prod(-1)).tolist()
    else:
        error_msg = "Unsupported visual model"
        raise NotImplementedError(error_msg)

    image_features = torch.split(image_features, image_num_patches, dim=0)

    if "LlavaNextForConditionalGeneration" in model.config.architectures:
        embed_std = 1 / np.sqrt(model.config.text_config.hidden_size)
        image_newline = torch.Tensor(
            torch.randn(image_features[0].shape[-1], dtype=image_features[0].dtype) * embed_std
        ).to(model.device)
        image_features, _ = model.pack_image_features(
            image_features,
            inputs.image_sizes,
            vision_feature_select_strategy=kwargs.get("vision_feature_select_strategy"),
            image_newline=image_newline,
        )
    elif (
        "Qwen2_5_VLForConditionalGeneration" in model.config.architectures
        or "Qwen2VLForConditionalGeneration" in model.config.architectures
    ):
        spatial_merge_size = model.visual.config.spatial_merge_size
        pooled_image_features = []
        for img_feat, (t, h, w) in zip(image_features, kwargs["image_grid_thw"]):
            num_patches, d = img_feat.shape
            assert t == 1, "Only single-frame temporal dimension supported"
            assert h * w == num_patches, f"H*W != num_patches: {h}*{w} != {num_patches}"

            # Reshape to [1, D, H, W]
            x = img_feat.view(h, w, d).permute(2, 0, 1).unsqueeze(0)

            # Apply avg pooling
            x_pooled = F.avg_pool2d(x, kernel_size=spatial_merge_size, stride=spatial_merge_size)

            # Reshape back to [num_pooled_patches, D]
            pooled = x_pooled.squeeze(0).permute(1, 2, 0).reshape(-1, d)
            pooled_image_features.append(pooled)
        image_features = pooled_image_features

    if image_features[0].dim() < 3:
        image_features = [feat_i.unsqueeze(0) for feat_i in image_features]
    return image_features


def get_cdpruner_mask(image_embeds, image_features, text_embeds, special_image_mask, num_keep_tokens, theta):
    """
    Generate a mask to retain image tokens based on fast MAP inference using Conditional DPP for token selection.
    """
    keep_indices = []
    offset = 0
    # Compute keep_indices for each image embedding
    for emb_i, feat_i in zip(image_embeds, image_features):
        rel_i = get_relevance_score(emb_i.unsqueeze(0), text_embeds)
        sim_i = get_visual_similarity(feat_i)
        kernel_i = build_conditional_kernel_matrix(rel_i, sim_i, theta)
        keep_i = conditional_dpp_map(kernel_i, num_keep_tokens)[0] + offset
        keep_indices.append(keep_i)
        offset += emb_i.shape[0]

    keep_indices = torch.cat(keep_indices, dim=0)

    # Get the positions of the selected image tokens
    image_token_positions = torch.nonzero(special_image_mask[0], as_tuple=False).squeeze(1)
    kept_positions = image_token_positions[keep_indices]

    # Build mask to keep: original text + selected image tokens
    kept_mask = ~special_image_mask
    kept_mask[0, kept_positions] = True

    return kept_mask


def extract_column_types_from_prompt(prompt: str) -> Dict[str, str]:
    """
    Extract column names and their types from prompt.
    Example: "{image:image_path}" -> {"image_path": "image"}
             "{text:question}" -> {"question": "text"}
    """
    pattern = r'\{(\w+):(\w+)\}'
    matches = re.findall(pattern, prompt)
    return {col_name: col_type for col_type, col_name in matches}

def encode_image_embedding_to_base64(embedding: torch.Tensor) -> str:
    """Convert tensor embedding to base64 string."""
    numpy_array = embedding.cpu().numpy()
    bytes_data = numpy_array.tobytes()
    base64_str = base64.b64encode(bytes_data).decode('utf-8')
    return base64_str

def extract_image_binary_from_pope_data(image_data):
    """Extract image binary from POPE data format."""
    if isinstance(image_data, (list, tuple)):
        return image_data[0] if len(image_data) > 0 else image_data
    return image_data

def sem_topk_impl(
    self: DataFrame,
    semantic_col: str or Column,
    instruction: str,
    k: int,
    group_by_cols: list[str or Column] = None,
    model: str = "/data/models/Qwen2.5-7B-Instruct/",
    llm_url: str = "http://localhost:8000/v1/chat/completions",
) -> DataFrame:
    spark = self.sparkSession
    jvm = spark._sc._jvm

    if isinstance(semantic_col, str):
        semantic_col_expr = pyspark_col(semantic_col)._jc.expr()
    elif isinstance(semantic_col, Column):
        semantic_col_expr = semantic_col._jc.expr()
    else:
        raise TypeError("semantic_col must be a string or a PySpark Column")

    scala_group_by_exprs_list = jvm.java.util.ArrayList()
    if group_by_cols:
        for gb_col_spec in group_by_cols:
            if isinstance(gb_col_spec, str):
                scala_group_by_exprs_list.add(pyspark_col(gb_col_spec)._jc.expr())
            elif isinstance(gb_col_spec, Column):
                scala_group_by_exprs_list.add(gb_col_spec._jc.expr())
            else:
                raise TypeError("group_by_cols must be a list of strings or PySpark Columns")

    child_logical_plan = self._jdf.logicalPlan()

    scala_helper_class_name = "com.probe.sparkai.utils.SemTopKPySparkHelper"
    helper = getattr(jvm, scala_helper_class_name)
    scala_result_df = helper.applySemTopK(
        spark._jsparkSession,
        child_logical_plan,
        semantic_col_expr,
        instruction,
        k,
        model,
        llm_url,
        scala_group_by_exprs_list
    )
    return DataFrame(scala_result_df, spark)


def getOriginalVisualToken(model, vision_tower, image_binary):
    
    preprocess_start = time.time()   
    image = Image.open(io.BytesIO(image_binary))
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    preprocess_end = time.time()   
    preprocess_time = preprocess_end - preprocess_start
    
    encode_begin = time.time()
    model_device = vision_tower.device

    image_forward_outs = vision_tower.vision_tower(
        images.to(device=model_device, dtype=vision_tower.dtype),
        output_hidden_states=True,
        output_attentions=True
    )
    image_outputs = vision_tower.feature_select(image_forward_outs)
    image_features = image_outputs.to(images.dtype)
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(model_device)
    image_features = model.multi_modal_projector(image_features).detach().cpu() 
    encode_end = time.time()
    encode_time = encode_end - encode_begin

    return image_features,preprocess_time,encode_time


def sem_map_impl(
    self: DataFrame,
    prompt: str,
    col_out: str,
    append_right: bool = False,
    do_reorder_columns: bool = False,
    do_reorder_rows: bool = False,
    url: str = "http://localhost:8000/v1/completions",
    batch_size: int = 32,
    model: str = "",
    max_tokens: int = 16,
    temperature: float = 1.0,
    verbose: bool = False,
    stop: List[str] = None,
    timeout: int = 300,
) -> DataFrame:
    if stop is None:
        stop = []
    if not hasattr(self, 'sparkSession') or self.sparkSession is None:
        raise RuntimeError(
            "SparkSession not available on DataFrame. "
            "This usually means the DataFrame was not created correctly or "
            "the extension is being used in an unsupported way."
        )

    if isinstance(prompt, str) and isinstance(col_out, str):
        # Call the Java semMap method with appropriate arguments
        try:
            jdf = self.sem_map_jdf(
                prompt,
                col_out,
                append_right,
                do_reorder_columns,
                do_reorder_rows,
                url,
                batch_size,
                model,
                max_tokens,
                temperature,
                verbose,
                stop,
                timeout,
            )
        except Exception as e:
            print(f"ERROR: Call to Java _jdf.semMap with parameters failed: {e}")
            raise e
    else:
        raise TypeError(
            f"Arguments for sem_map_impl must be strings or compatible types. "
            f"Got types: prompt={type(prompt)}, col_out={type(col_out)}"
        )

    return DataFrame(jdf, self.sparkSession)


def sem_join_impl(
    self: DataFrame,
    right: DataFrame,
    prompt: str,
    true_criteria: str = "True",
    false_criteria: str = "",
    prompt_prefix: str = DEFAULT_SYSTEM_PROMPT,
    prompt_suffix: str = "\n[Answer Format]:\nThe answer should only be either \"True\" or \"False\", give your answer first then reason.\nAnswer:",
    lower: bool = False,
    append_right: bool = False,
    do_reorder_columns: bool = False,
    do_reorder_rows: bool = False,
    url: str = "http://localhost:8000/v1/chat/completions",
    batch_size: int = 32,
    model: str = "",
    embedding_model: str = "",
    max_tokens: int = 16,
    temperature: float = 1.0,
    optimized: bool = False,
    verbose: bool = False,
    stop: List[str] = None,
    timeout: int = 300,
    calibrate: bool = True,
    device: str = "cuda",
    gpu_id: int = 0,
    recall_target: float = 0.8,
    failure_probability: float = 0.2,
    precision_target: float = 0.8,
    sim_join_only: bool = False,
    map_prompt: str = "Given {{left_on}}, try to infer or extract the most relevant {right_on}. Answer as a list of 2-10 comma-separated keywords or short phrases for {right_on}. Give your answer first. Answer: ",
    distributed: bool = False,
    left_embedding_col: str = "",
    right_embedding_col: str = "",
    confidence_level: float = 0.95,
    error_tolerance: float = 1.0,
    is_approximate: bool = False,
    batch_prompt_size: int = 10,
    embedding_pre_screen: bool = False,
    eer_sample: bool = False,
    cwre: bool = False,
    adaptive_batch_size: bool = False,
    consistency_sample: bool = False,
    pilot_ratio: float = 0.016,
    adaptive_execution: bool = False,
    proxy_type: str = "auto_detect",
    ocr_model: str = "",
    yolo_model: str = "",
    max_concurrent_requests: int = 256,
    proxy_keep_ratio: float = 0.25,
    bound_type: str = "default",
    adaptive_threshold: bool = False,
    embedding_device: str = "cuda",
    proxy_sample: bool = False,
    sampling_method: str = "random",
    sample_size: int = 500,
    imbalance_strategy: str = "weighted",
    confidence_threshold: float = 0.5,
    proxy_model_path: str = "",
    refine_batch_size: int = 32,
    max_refine_rounds: int = 100,
    proxy_warm_start: bool = False,
    topk_prefilter: bool = False,
    topk_k: int = 0,
    exclude_self_pairs: bool = False,
    join_sim_mode: str = "pair",
    lr_feature_mode: str = "embedding",
    budget_size: int = 0,
    cluster_proxy_mode: str = "propagation",
    n_clusters: int = 0,
    knn_k: int = 0,
    n_strata: int = 10,
    bas_pilot_ratio: float = 0.15,
    allocation_method: str = "neyman",
    reg_method: str = "exponential",
    reg_strength: float = 0.5,
    auto_k: bool = False,
    refine: bool = False,
    diffusion_alpha: float = 0.8,
    propagation_beta: float = 0.8,
    refine_num_iter: int = 50,
    refine_tol: float = 1e-4,
    refine_patience: int = 5,
    refine_enabled: bool = True,
    activeSampling_B1: int = 5,
    activeSampling_vs_ratio: float = 0.5,
    activeSampling_n_modes: int = 1,
    activeSampling_k_intervals: Optional[int] = None,
    activeSampling_k_auto: bool = False,
    activeSampling_k_max: int = 4,
    activeSampling_k_gain_tol: float = 0.02,
    activeSampling_class_balance: bool = False,
    activeSampling_n_phases: int = 1,
    activeSampling_lr_blend_schedule: str = "linear",
    activeSampling_cold_start_phases: int = 1,
    activeSampling_lr_explore_frac: float = 0.0,
    activeSampling_phase_budget_split: str = "equal",
    defer_enabled: bool = False,
    verify_uncertain: bool = True,
    router_max_calls: Optional[int] = None,
    decision_mode: str = "force_label",
    deferral_features: Optional[Sequence[str]] = None,
    deferral_rank_axis: str = "blended",
    deferral_hidden_dims_grid: Optional[Sequence[Sequence[int]]] = None,
    deferral_alpha_grid: Optional[Sequence[float]] = None,
    deferral_k_folds: int = 5,
    deferral_inner_cv_k: int = 3,
    deferral_dagger_beta: float = 0.00,
    deferral_min_inner_cv_auc: float = 0.60,
    deferral_min_probe_auc: float = 0.60,
    lr_C: float = 1.0,
    base_threshold: Optional[float] = None,
    lr_class_weight_override: Optional[str] = None,
    random_seed: Optional[int] = 42,
    viz_tag: str = "",
) -> DataFrame:
    if stop is None:
        stop = []

    if not hasattr(self, 'sparkSession') or self.sparkSession is None:
        raise RuntimeError(
            "SparkSession not available on DataFrame. "
            "This usually means the DataFrame was not created correctly or"
            "the extension is being used in an unsupported way."
        )

    if proxy_sample:
        from proxy_model import proxy_join
        return proxy_join(
            left_df=self,
            right_df=right,
            prompt=prompt,
            model=model,
            url=url,
            embedding_model=embedding_model,
            sample_size=sample_size,
            sampling_method=sampling_method,
            imbalance_strategy=imbalance_strategy,
            proxy_model_path=proxy_model_path,
            proxy_warm_start=proxy_warm_start,
            max_tokens=max_tokens,
            batch_size=batch_size,
            verbose=verbose,
            max_concurrent_requests=max_concurrent_requests,
            true_criteria=true_criteria,
            refine_batch_size=refine_batch_size,
            max_refine_rounds=max_refine_rounds,
            confidence_threshold=confidence_threshold,
            topk_prefilter=topk_prefilter,
            topk_k=topk_k,
            exclude_self_pairs=exclude_self_pairs,
            join_sim_mode=join_sim_mode,
            lr_feature_mode=lr_feature_mode,
            budget_size=budget_size if budget_size > 0 else None,
            cluster_proxy_mode=cluster_proxy_mode,
            n_clusters=n_clusters,
            knn_k=knn_k,
            n_strata=n_strata,
            bas_pilot_ratio=bas_pilot_ratio,
            allocation_method=allocation_method,
            reg_method=reg_method,
            reg_strength=reg_strength,
            auto_k=auto_k,
            refine=refine,
            diffusion_alpha=diffusion_alpha,
            propagation_beta=propagation_beta,
            refine_num_iter=refine_num_iter,
            refine_tol=refine_tol,
            refine_patience=refine_patience,
            refine_enabled=refine_enabled,
            activeSampling_B1=activeSampling_B1,
            activeSampling_vs_ratio=activeSampling_vs_ratio,
            activeSampling_n_modes=activeSampling_n_modes,
            activeSampling_k_intervals=activeSampling_k_intervals,
            activeSampling_k_auto=activeSampling_k_auto,
            activeSampling_k_max=activeSampling_k_max,
            activeSampling_k_gain_tol=activeSampling_k_gain_tol,
            activeSampling_class_balance=activeSampling_class_balance,
            activeSampling_n_phases=activeSampling_n_phases,
            activeSampling_lr_blend_schedule=activeSampling_lr_blend_schedule,
            activeSampling_cold_start_phases=activeSampling_cold_start_phases,
            activeSampling_lr_explore_frac=activeSampling_lr_explore_frac,
            activeSampling_phase_budget_split=activeSampling_phase_budget_split,
            defer_enabled=defer_enabled,
            verify_uncertain=verify_uncertain,
            router_max_calls=router_max_calls,
            decision_mode=decision_mode,
            deferral_features=deferral_features,
            deferral_rank_axis=deferral_rank_axis,
            deferral_hidden_dims_grid=deferral_hidden_dims_grid,
            deferral_alpha_grid=deferral_alpha_grid,
            deferral_k_folds=deferral_k_folds,
            deferral_inner_cv_k=deferral_inner_cv_k,
            deferral_dagger_beta=deferral_dagger_beta,
            deferral_min_inner_cv_auc=deferral_min_inner_cv_auc,
            deferral_min_probe_auc=deferral_min_probe_auc,
            lr_C=lr_C,
            base_threshold=base_threshold,
            lr_class_weight_override=lr_class_weight_override,
            random_seed=random_seed,
            viz_tag=viz_tag,
        )

    # When is_approximate=True, delegate to sem_approximate_join which uses
    # the dedicated AQP path (ApproximateJoin logical plan → Rust AQP engine).
    if is_approximate:
        return sem_approximate_join_impl(
            self,
            right=right,
            prompt=prompt,
            url=url,
            model=model,
            batch_size=batch_size,
            max_tokens=max_tokens,
            timeout=timeout,
            batch_prompt_size=batch_prompt_size,
            confidence_level=confidence_level,
            error_tolerance=error_tolerance,
            verbose=verbose,
            embedding_pre_screen=embedding_pre_screen,
            eer_sample=eer_sample,
            cwre=cwre,
            adaptive_batch_size=adaptive_batch_size,
            consistency_sample=consistency_sample,
            embedding_model=embedding_model,
            pilot_ratio=pilot_ratio,
            proxy_keep_ratio=proxy_keep_ratio,
            adaptive_execution=adaptive_execution,
            proxy_type=proxy_type,
            ocr_model=ocr_model,
            yolo_model=yolo_model,
            bound_type=bound_type,
            adaptive_threshold=adaptive_threshold,
            embedding_device=embedding_device,
        )

    try:
        jdf = self.sem_join_jdf(
            right._jdf,
            prompt,
            true_criteria,
            false_criteria,
            prompt_prefix,
            prompt_suffix,
            lower,
            append_right,
            do_reorder_columns,
            do_reorder_rows,
            url,
            batch_size,
            model,
            embedding_model,
            max_tokens,
            temperature,
            optimized,
            verbose,
            stop,
            timeout,
            calibrate,
            device,
            gpu_id,
            recall_target,
            failure_probability,
            precision_target,
            sim_join_only,
            map_prompt,
            distributed,
            left_embedding_col,
            right_embedding_col,
            confidence_level,
            error_tolerance,
            False,
            max_concurrent_requests,
            float(proxy_keep_ratio),
            proxy_sample,
            sampling_method,
            sample_size,
            imbalance_strategy,
            confidence_threshold,
            proxy_model_path,
            proxy_warm_start,
        )
    except Exception as e:
        print(f"ERROR: Call to Java _jdf.semJoin with parameters failed: {e}")
        raise e

    return DataFrame(jdf, self.sparkSession)


def sem_sim_join_impl(
    self: DataFrame,
    right: DataFrame,
    prompt: str,
    col_out: str = "__distance",
    batch_size: int = 32,
    embedding_model: str = "",
    verbose: bool = False,
    calibrate: bool = True,
    device: str = "cuda",
    gpu_id: int = 0,
    metric_type_str: str = "cosine",
    k: int = -1,
    left_embedding_col: str = "",
    right_embedding_col: str = "",
) -> DataFrame:
    if not hasattr(self, 'sparkSession') or self.sparkSession is None:
        raise RuntimeError(
            "SparkSession not available on DataFrame. "
            "This usually means the DataFrame was not created correctly or "
            "the extension is being used in an unsupported way."
        )

    try:
        # Convert parameter names to match Java method signature
        jdf = self.sem_sim_join_jdf(
            right._jdf,
            prompt,
            col_out,
            batch_size,
            embedding_model,
            verbose,
            calibrate,
            device,
            gpu_id,
            metric_type_str,
            k,
            left_embedding_col,
            right_embedding_col,
        )
    except Exception as e:
        print(f"ERROR: Call to Java _jdf.semSimJoin with parameters failed: {e}")
        raise e

    return DataFrame(jdf, self.sparkSession)


def sem_sim_filter_impl(
    self: DataFrame,
    prompt: str,
    query: str = "",
    threshold: float = 0.5,
    col_out: str = "__distance",
    embedding_model: str = "",
    embedding_url: str = "",
    timeout: int = 300,
    verbose: bool = False,
    return_distance: bool = False,
) -> DataFrame:
    if not hasattr(self, 'sparkSession') or self.sparkSession is None:
        raise RuntimeError(
            "SparkSession not available on DataFrame. "
            "This usually means the DataFrame was not created correctly or "
            "the extension is being used in an unsupported way."
        )

    try:
        jdf = self.sem_sim_filter_jdf(
            prompt,
            query,
            threshold,
            col_out,
            embedding_model,
            embedding_url,
            timeout,
            verbose,
            return_distance,
        )
    except Exception as e:
        print(f"ERROR: Call to Java _jdf.semSimFilter with parameters failed: {e}")
        raise e

    return DataFrame(jdf, self.sparkSession)


def sem_embed_impl(
    self: DataFrame,
    prompt: str,
    col_out: str = "__embedding",
    normalization: bool = True,
    batch_size: int = 32,
    embedding_model: str = "",
    device: str = "cuda",
    gpu_id: int = 0,
    verbose: bool = False,
) -> DataFrame:
    if not hasattr(self, 'sparkSession') or self.sparkSession is None:
        raise RuntimeError(
            "SparkSession not available on DataFrame. "
            "This usually means the DataFrame was not created correctly or "
            "the extension is being used in an unsupported way."
        )

    try:
        # Convert parameter names to match Java method signature
        jdf = self.sem_embed_jdf(
            prompt,
            col_out,
            normalization,
            batch_size,
            embedding_model,
            device,
            gpu_id,
            verbose,
        )
    except Exception as e:
        print(f"ERROR: Call to Java _jdf.semEmbed with parameters failed: {e}")
        raise e

    return DataFrame(jdf, self.sparkSession)

def sem_groupby_impl(
    self: DataFrame,
    prompt: str,
    proxy_model: str,
    proxy_url: str,
    main_model: str,
    main_url: str,
    embedder_model:str ,
    enableColReorder : bool, 
    enableRowReorder : bool,
    num_clusters: int = 1,
    max_tokens: int=100
) -> DataFrame:
    if not hasattr(self, 'sparkSession') or self.sparkSession is None:
        raise RuntimeError(
            "SparkSession not available on DataFrame. "
            "This usually means the DataFrame was not created correctly or "
            "the extension is being used in an unsupported way."
        )
    if isinstance(prompt, str):
        # Call the Java sem_groupby method with a String argument
        try:
            print("PROMPT in sem ops register ++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(prompt)
            print("URL in sem ops register ++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(main_url)
            jdf = self.sem_groupby_jdf(prompt, proxy_model, proxy_url, main_model, main_url, embedder_model, enableColReorder, enableRowReorder, num_clusters, max_tokens)
        except Exception as e:
            print(f"ERROR: Call to Java _jdf.semGroupBy('{prompt}') failed.")
            print("Ensure 'semGroupBy(String)' is defined on your Spark's Java/Scala DataFrame (Dataset) class.")
            raise e
    else:
        raise TypeError(
            f"Condition for sem_groupby must be a PySpark Column or a SQL string expression. "
            f"Got type: {type(prompt)}"
        )
    return DataFrame(jdf, self.sparkSession)



def sem_agg_impl(
    self: DataFrame,
    prompt: str,
    model: str ="/data/models/Qwen2.5-7B-Instruct",
    url: str="http://localhost:8007/v1/completions",
    groupingColumn: str = "grouped_category",
    enableColumnSorting: bool = False, 
    enableRowSorting: bool = False, 
    max_tokens: int = 100, 
    model_token_limit: int = 32768
) -> DataFrame:
    if not hasattr(self, 'sparkSession') or self.sparkSession is None:
        raise RuntimeError(
            "SparkSession not available on DataFrame. "
            "This usually means the DataFrame was not created correctly or "
            "the extension is being used in an unsupported way."
        )

    if isinstance(prompt, str):
        # Call the Java sem_agg method with a String argument
        try:
            jdf = self.sem_agg_jdf(prompt, model, url, groupingColumn, enableColumnSorting, enableRowSorting, max_tokens, model_token_limit)
        except Exception as e:
            print(f"ERROR: Call to Java _jdf.semAgg('{prompt}') failed.")
            print("Ensure 'semAgg(String)' is defined on your Spark's Java/Scala DataFrame (Dataset) class.")
            raise e
    else:
        raise TypeError(
            f"Condition for sem_agg must be a PySpark Column or a SQL string expression. "
            f"Got type: {type(prompt)}"
        )
    return DataFrame(jdf, self.sparkSession)


def sem_filter_impl(
    self: DataFrame,
    prompt: str,
    model: str,
    url: str,
    max_tokens: int=100,
    enableColReorder: bool = False,
    enableRowReorder: bool = False,
    use_cascade: bool = False,
    proxy_model: str = None,
    proxy_url: str = None,
    batch_size: int = 32,
    stop: List[str] = None,
    timeout: int = 300,
    num_quantiles: int = 50,
    recall_target: float = 0.8,
    failure_probability: float = 0.2,
    precision_target: float = 0.8,
    confidence_level: float = 0.95,
    error_tolerance: float = 1.0,
    is_approximate: bool = False,
    verbose: bool = False,
    embedding_pre_screen: bool = False,
    eer_sample: bool = False,
    cwre: bool = False,
    adaptive_batch_size: bool = False,
    consistency_sample: bool = False,
    embedding_model: str = "",
    pilot_ratio: float = 0.10,
    proxy_keep_ratio: float = 0.25,
    adaptive_execution: bool = False,
    batch_prompt_size: int = 10,
    proxy_type: str = "auto_detect",
    max_concurrent_requests: int = 256,
    bound_type: str = "default",
    adaptive_threshold: bool = False,
    proxy_sample: bool = False,
    sampling_method: str = "random",
    sample_size: int = 200,
    imbalance_strategy: str = "weighted",
    confidence_threshold: float = 0.5,
    proxy_model_path: str = "",
    refine_batch_size: int = 32,
    max_refine_rounds: int = 100,
    proxy_warm_start: bool = False,
    budget_size: int = 0,
    cluster_proxy_mode: str = "propagation",
    n_clusters: int = 0,
    knn_k: int = 0,
    explore_ratio: float = 0.4,
    boundary_delta: float = 0.05,
    lr_feature_mode: str = "embedding",
    n_strata: int = 10,
    bas_pilot_ratio: float = 0.15,
    allocation_method: str = "neyman",
    reg_method: str = "exponential",
    reg_strength: float = 0.5,
    auto_k: bool = False,
    refine: bool = False,
    diffusion_alpha: float = 0.8,
    propagation_beta: float = 0.8,
    refine_num_iter: int = 50,
    refine_tol: float = 1e-4,
    refine_patience: int = 5,
    refine_enabled: bool = True,
    activeSampling_B1: int = 5,
    activeSampling_vs_ratio: float = 0.5,
    activeSampling_n_modes: int = 1,
    activeSampling_k_intervals: Optional[int] = None,
    activeSampling_k_auto: bool = False,
    activeSampling_k_max: int = 4,
    activeSampling_k_gain_tol: float = 0.02,
    activeSampling_class_balance: bool = False,
    activeSampling_n_phases: int = 1,
    activeSampling_lr_blend_schedule: str = "linear",
    activeSampling_cold_start_phases: int = 1,
    activeSampling_lr_explore_frac: float = 0.0,
    activeSampling_phase_budget_split: str = "equal",
    defer_enabled: bool = False,
    verify_uncertain: bool = True,
    router_max_calls: Optional[int] = None,
    decision_mode: str = "force_label",
    deferral_features: Optional[Sequence[str]] = None,
    deferral_rank_axis: str = "blended",
    deferral_hidden_dims_grid: Optional[Sequence[Sequence[int]]] = None,
    deferral_alpha_grid: Optional[Sequence[float]] = None,
    deferral_k_folds: int = 5,
    deferral_inner_cv_k: int = 3,
    deferral_dagger_beta: float = 0.00,
    deferral_min_inner_cv_auc: float = 0.60,
    deferral_min_probe_auc: float = 0.60,
    lr_C: float = 1.0,
    base_threshold: Optional[float] = None,
    lr_class_weight_override: Optional[str] = None,
    random_seed: Optional[int] = 42,
    viz_tag: str = "",
) -> DataFrame:
    if stop is None:
        stop = []

    if not hasattr(self, 'sparkSession') or self.sparkSession is None:
        raise RuntimeError(
            "SparkSession not available on DataFrame. "
            "This usually means the DataFrame was not created correctly or "
            "the extension is being used in an unsupported way."
        )

    if proxy_sample:
        from proxy_model import proxy_filter
        return proxy_filter(
            df=self,
            prompt=prompt,
            model=model,
            url=url,
            embedding_model=embedding_model,
            sample_size=sample_size,
            sampling_method=sampling_method,
            imbalance_strategy=imbalance_strategy,
            proxy_model_path=proxy_model_path,
            proxy_warm_start=proxy_warm_start,
            max_tokens=max_tokens,
            batch_size=batch_size,
            verbose=verbose,
            refine_batch_size=refine_batch_size,
            max_refine_rounds=max_refine_rounds,
            max_concurrent_requests=max_concurrent_requests,
            confidence_threshold=confidence_threshold,
            budget_size=budget_size if budget_size > 0 else None,
            cluster_proxy_mode=cluster_proxy_mode,
            n_clusters=n_clusters,
            knn_k=knn_k,
            lr_feature_mode=lr_feature_mode,
            n_strata=n_strata,
            bas_pilot_ratio=bas_pilot_ratio,
            allocation_method=allocation_method,
            reg_method=reg_method,
            reg_strength=reg_strength,
            auto_k=auto_k,
            refine=refine,
            diffusion_alpha=diffusion_alpha,
            propagation_beta=propagation_beta,
            refine_num_iter=refine_num_iter,
            refine_tol=refine_tol,
            refine_patience=refine_patience,
            refine_enabled=refine_enabled,
            activeSampling_B1=activeSampling_B1,
            activeSampling_vs_ratio=activeSampling_vs_ratio,
            activeSampling_n_modes=activeSampling_n_modes,
            activeSampling_k_intervals=activeSampling_k_intervals,
            activeSampling_k_auto=activeSampling_k_auto,
            activeSampling_k_max=activeSampling_k_max,
            activeSampling_k_gain_tol=activeSampling_k_gain_tol,
            activeSampling_class_balance=activeSampling_class_balance,
            activeSampling_n_phases=activeSampling_n_phases,
            activeSampling_lr_blend_schedule=activeSampling_lr_blend_schedule,
            activeSampling_cold_start_phases=activeSampling_cold_start_phases,
            activeSampling_lr_explore_frac=activeSampling_lr_explore_frac,
            activeSampling_phase_budget_split=activeSampling_phase_budget_split,
            defer_enabled=defer_enabled,
            verify_uncertain=verify_uncertain,
            router_max_calls=router_max_calls,
            decision_mode=decision_mode,
            deferral_features=deferral_features,
            deferral_rank_axis=deferral_rank_axis,
            deferral_hidden_dims_grid=deferral_hidden_dims_grid,
            deferral_alpha_grid=deferral_alpha_grid,
            deferral_k_folds=deferral_k_folds,
            deferral_inner_cv_k=deferral_inner_cv_k,
            deferral_dagger_beta=deferral_dagger_beta,
            deferral_min_inner_cv_auc=deferral_min_inner_cv_auc,
            deferral_min_probe_auc=deferral_min_probe_auc,
            lr_C=lr_C,
            base_threshold=base_threshold,
            lr_class_weight_override=lr_class_weight_override,
            random_seed=random_seed,
            viz_tag=viz_tag,
        )

    if isinstance(prompt, str):
        try:
            if use_cascade:
                # In cascade mode: model/url = oracle (expensive), proxy_model/proxy_url = proxy (cheap)
                if proxy_model is None or proxy_url is None:
                    raise ValueError(
                        "When use_cascade=True, both proxy_model and proxy_url must be provided. "
                        "The 'model' and 'url' parameters will be used as the oracle (expensive model), "
                        "while 'proxy_model' and 'proxy_url' will be used as the proxy (cheap model)."
                    )
                jdf = self.sem_filter_cascaded_jdf(
                    prompt, model, url, proxy_model, proxy_url,
                    batch_size, max_tokens,
                    stop, timeout, num_quantiles, recall_target,
                    failure_probability, precision_target, verbose
                )
            else:
                jdf = self.sem_filter_jdf(prompt, model, url, max_tokens, enableColReorder, enableRowReorder, confidence_level, error_tolerance, is_approximate, batch_size, embedding_pre_screen, eer_sample, cwre, adaptive_batch_size, consistency_sample, embedding_model, pilot_ratio, proxy_keep_ratio, adaptive_execution, batch_prompt_size, proxy_type, max_concurrent_requests, verbose, bound_type, adaptive_threshold, proxy_sample, sampling_method, sample_size, imbalance_strategy, confidence_threshold, proxy_model_path, proxy_warm_start)
        except Exception as e:
            print(f"ERROR: Call to Java _jdf.semFilter('{prompt}') failed.")
            print("Ensure 'semFilter(String)' is defined on your Spark's Java/Scala DataFrame (Dataset) class.")
            raise e
    else:
        raise TypeError(
            f"Condition for sem_filter must be a PySpark Column or a SQL string expression. "
            f"Got type: {type(prompt)}"
        )
    return DataFrame(jdf, self.sparkSession)


def sem_filter_jdf_impl(self, prompt, model, url, max_tokens, enableColReorder, enableRowReorder, confidence_level, error_tolerance, is_approximate, batch_size, embedding_pre_screen, eer_sample, cwre, adaptive_batch_size, consistency_sample, embedding_model, pilot_ratio, proxy_keep_ratio=0.25, adaptive_execution=False, batch_prompt_size=10, proxy_type="auto_detect", max_concurrent_requests=256, verbose=False, bound_type="default", adaptive_threshold=False, proxy_sample=False, sampling_method="random", sample_size=200, imbalance_strategy="weighted", confidence_threshold=0.5, proxy_model_path="", proxy_warm_start=False):
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAIFilter(self._jdf, prompt, model, url,
                                                                                 max_tokens, enableColReorder, enableRowReorder,
                                                                                 confidence_level, error_tolerance, is_approximate, batch_size,
                                                                                 embedding_pre_screen, eer_sample, cwre, adaptive_batch_size, consistency_sample, embedding_model, pilot_ratio, float(proxy_keep_ratio), adaptive_execution, batch_prompt_size,
                                                                                 max_concurrent_requests, bool(verbose), proxy_type, bound_type, bool(adaptive_threshold),
                                                                                 bool(proxy_sample), sampling_method, int(sample_size), imbalance_strategy, float(confidence_threshold), proxy_model_path, bool(proxy_warm_start))


def sem_filter_cascaded_jdf_impl(self, prompt, model, url, proxy_model, proxy_url,
                                   batch_size, max_tokens,
                                   stop, timeout, num_quantiles, recall_target,
                                   failure_probability, precision_target, verbose):
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAIFilterCascaded(
        self._jdf, prompt, model, url, proxy_model, proxy_url,
        batch_size, max_tokens,
        stop, timeout, num_quantiles, recall_target,
        failure_probability, precision_target, verbose
    )


def sem_approximate_filter_impl(
    self: DataFrame,
    prompt: str,
    url: str = "http://localhost:8000/v1/chat/completions",
    model: str = "default",
    batch_size: int = 100,
    max_tokens: int = 512,
    timeout: int = 600,
    batch_prompt_size: int = 10,
    confidence_level: float = 0.95,
    error_tolerance: float = 0.1,
    verbose: bool = False,
    embedding_pre_screen: bool = False,
    eer_sample: bool = False,
    cwre: bool = False,
    adaptive_batch_size: bool = False,
    consistency_sample: bool = False,
    embedding_model: str = "",
    pilot_ratio: float = 0.10,
    proxy_keep_ratio: float = 0.25,
    adaptive_execution: bool = False,
    proxy_type: str = "auto_detect",
    bound_type: str = "default",
    adaptive_threshold: bool = False,
) -> DataFrame:
    """
    Apply approximate semantic filter with batch prompting.
    Delegates to sem_filter with is_approximate=True to use a unified callstack.
    Implements ThalamusDB-style approximate query processing:
    - For retrieval queries: shows rows that appear in ALL possible results (intersection)
    - Error calculation: max_rows/intersection_rows - 1 (0 if both are 0)

    Args:
        prompt: Filter condition with column placeholders like {column_name}
        url: LLM API endpoint URL
        model: LLM model name
        batch_size: Number of rows to process per partition
        max_tokens: Maximum tokens for LLM response
        timeout: Request timeout in seconds
        batch_prompt_size: Number of records per batch prompt (for batch prompting)
        confidence_level: Confidence target for probabilistic guarantees
        error_tolerance: Maximum acceptable retrieval/aggregate error
        verbose: Enable verbose logging
        embedding_pre_screen: Use embedding similarity to pre-classify rows
        eer_sample: Use Expected Error Reduction sampling
        cwre: Use Confidence-Weighted Recall Error metric
        adaptive_batch_size: Use adaptive batch sizing based on query difficulty
        consistency_sample: Use consistency-aware sampling with re-evaluation
        embedding_model: Path to the embedding model directory (empty string disables embedding)
        pilot_ratio: Fraction of total rows for the pilot phase (0.016 = ~1.6%). 0 disables pilot.
        proxy_keep_ratio: Fraction of rows to keep from proxy model results (default 0.25).
        adaptive_execution: Enable adaptive chunk sizing and auto error tolerance tuning.
        proxy_type: Proxy scoring strategy. Options:
            "auto_detect" - legacy auto: SigLIP if image columns, else text_only
            "auto" - adaptive two-stage selection (modality detection + empirical probe)
            "auto_llm" - adaptive two-stage selection with LLM-guided proxy choice
            "text_only" - MiniLM cosine similarity (default lightweight)
            "bm25" - BM25 lexical keyword matching (best for exact name/keyword lookups)
            "bge" - BGE-base-v1.5 dense embeddings (best for semantic similarity)
            "e5" - E5-base-v2 dense embeddings (good for passage retrieval)
            "siglip" - SigLIP zero-shot image-text matching
            "ocr" - OCR + fuzzy text matching (best for text-in-image queries)

    Returns:
        DataFrame with approximate filter applied
    """
    return sem_filter_impl(
        self,
        prompt=prompt,
        model=model,
        url=url,
        max_tokens=max_tokens,
        confidence_level=confidence_level,
        error_tolerance=error_tolerance,
        is_approximate=True,
        batch_size=batch_size,
        embedding_pre_screen=embedding_pre_screen,
        eer_sample=eer_sample,
        cwre=cwre,
        adaptive_batch_size=adaptive_batch_size,
        consistency_sample=consistency_sample,
        embedding_model=embedding_model,
        pilot_ratio=pilot_ratio,
        proxy_keep_ratio=proxy_keep_ratio,
        adaptive_execution=adaptive_execution,
        proxy_type=proxy_type,
        bound_type=bound_type,
        adaptive_threshold=adaptive_threshold,
    )


def sem_approximate_filter_jdf_impl(self, prompt, url, model, batch_size, max_tokens, timeout, batch_prompt_size, confidence_level, error_tolerance, verbose, embedding_pre_screen, eer_sample, cwre, adaptive_batch_size, consistency_sample, embedding_model, pilot_ratio, proxy_keep_ratio=0.25, adaptive_execution=False, proxy_type="auto_detect", bound_type="default", adaptive_threshold=False):
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAIApproximateFilter(
        self._jdf, prompt, url, model, batch_size, max_tokens, timeout, batch_prompt_size,
        confidence_level, error_tolerance, verbose,
        embedding_pre_screen, eer_sample, cwre, adaptive_batch_size, consistency_sample, embedding_model, pilot_ratio, proxy_keep_ratio, adaptive_execution, proxy_type, bound_type, bool(adaptive_threshold)
    )


def sem_batch_filter_impl(
    self: DataFrame,
    prompt: str,
    true_criteria: str = "true",
    false_criteria: str = "false",
    url: str = "http://localhost:8000/v1/chat/completions",
    model: str = "default",
    batch_size: int = 100,
    max_tokens: int = 512,
    timeout: int = 600,
    batch_prompt_size: int = 10,
    verbose: bool = False
) -> DataFrame:
    """
    Apply batch prompting semantic filter.
    Uses batch prompting to reduce LLM API calls by processing multiple rows per call.

    Args:
        prompt: Filter condition with column placeholders like {column_name}
        true_criteria: String that indicates match (default "true")
        false_criteria: String that indicates no match (default "false")
        url: LLM API endpoint URL
        model: LLM model name
        batch_size: Number of rows to process per partition
        max_tokens: Maximum tokens for LLM response
        timeout: Request timeout in seconds
        batch_prompt_size: Number of records per batch prompt
        verbose: Enable verbose logging

    Returns:
        DataFrame with batch filter applied
    """
    if not hasattr(self, 'sparkSession') or self.sparkSession is None:
        raise RuntimeError("SparkSession not available on DataFrame.")

    try:
        jdf = self.sem_batch_filter_jdf(
            prompt, true_criteria, false_criteria, url, model, batch_size,
            max_tokens, timeout, batch_prompt_size, verbose
        )
    except Exception as e:
        print(f"ERROR: Call to Java _jdf.semBatchFilter failed: {e}")
        raise e

    return DataFrame(jdf, self.sparkSession)


def sem_batch_filter_jdf_impl(self, prompt, true_criteria, false_criteria, url, model, batch_size, max_tokens, timeout, batch_prompt_size, verbose):
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAIBatchFilter(
        self._jdf, prompt, true_criteria, false_criteria, url, model, batch_size,
        max_tokens, timeout, batch_prompt_size, verbose
    )


def sem_approximate_join_impl(
    self: DataFrame,
    right: DataFrame,
    prompt: str,
    url: str = "http://localhost:8000/v1/chat/completions",
    model: str = "default",
    batch_size: int = 100,
    max_tokens: int = 512,
    timeout: int = 600,
    batch_prompt_size: int = 10,
    confidence_level: float = 0.95,
    error_tolerance: float = 0.1,
    verbose: bool = False,
    embedding_pre_screen: bool = False,
    eer_sample: bool = False,
    cwre: bool = False,
    adaptive_batch_size: bool = False,
    consistency_sample: bool = False,
    embedding_model: str = "",
    pilot_ratio: float = 0.10,
    proxy_keep_ratio: float = 0.25,
    adaptive_execution: bool = False,
    proxy_type: str = "auto_detect",
    ocr_model: str = "",
    yolo_model: str = "",
    bound_type: str = "default",
    adaptive_threshold: bool = False,
    embedding_device: str = "cuda",
) -> DataFrame:
    if not hasattr(self, 'sparkSession') or self.sparkSession is None:
        raise RuntimeError("SparkSession not available on DataFrame.")

    try:
        jdf = self.sem_approximate_join_jdf(
            right._jdf, prompt, url, model, batch_size, max_tokens, timeout, batch_prompt_size,
            confidence_level, error_tolerance, verbose,
            embedding_pre_screen, eer_sample, cwre, adaptive_batch_size, consistency_sample, embedding_model, pilot_ratio, proxy_keep_ratio, adaptive_execution, proxy_type, ocr_model, yolo_model, bound_type, adaptive_threshold, embedding_device
        )
    except Exception as e:
        print(f"ERROR: Call to Java _jdf.semApproximateJoin failed: {e}")
        raise e

    return DataFrame(jdf, self.sparkSession)


def sem_approximate_join_jdf_impl(self, right_jdf, prompt, url, model, batch_size, max_tokens, timeout, batch_prompt_size, confidence_level, error_tolerance, verbose, embedding_pre_screen, eer_sample, cwre, adaptive_batch_size, consistency_sample, embedding_model, pilot_ratio, proxy_keep_ratio=0.25, adaptive_execution=False, proxy_type="auto_detect", ocr_model="", yolo_model="", bound_type="default", adaptive_threshold=False, embedding_device="cuda"):
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAIApproximateJoin(
        self._jdf, right_jdf, prompt, url, model, batch_size, max_tokens, timeout, batch_prompt_size,
        confidence_level, error_tolerance, verbose,
        embedding_pre_screen, eer_sample, cwre, adaptive_batch_size, consistency_sample, embedding_model, pilot_ratio, proxy_keep_ratio, adaptive_execution, proxy_type, ocr_model, yolo_model, bound_type, bool(adaptive_threshold), embedding_device
    )


def sem_batch_join_impl(
    self: DataFrame,
    right: DataFrame,
    prompt: str,
    true_criteria: str = "true",
    false_criteria: str = "false",
    url: str = "http://localhost:8000/v1/chat/completions",
    model: str = "default",
    batch_size: int = 100,
    max_tokens: int = 512,
    timeout: int = 600,
    batch_prompt_size: int = 10,
    verbose: bool = False
) -> DataFrame:
    """
    Apply batch prompting semantic join.
    Uses batch prompting to reduce LLM API calls by processing multiple pairs per call.

    Args:
        right: Right DataFrame to join with
        prompt: Join condition with column placeholders
        true_criteria: String that indicates match (default "true")
        false_criteria: String that indicates no match (default "false")
        url: LLM API endpoint URL
        model: LLM model name
        batch_size: Number of rows to process per partition
        max_tokens: Maximum tokens for LLM response
        timeout: Request timeout in seconds
        batch_prompt_size: Number of pairs per batch prompt
        verbose: Enable verbose logging

    Returns:
        DataFrame with batch join applied
    """
    if not hasattr(self, 'sparkSession') or self.sparkSession is None:
        raise RuntimeError("SparkSession not available on DataFrame.")

    try:
        jdf = self.sem_batch_join_jdf(
            right._jdf, prompt, true_criteria, false_criteria, url, model,
            batch_size, max_tokens, timeout, batch_prompt_size, verbose
        )
    except Exception as e:
        print(f"ERROR: Call to Java _jdf.semBatchJoin failed: {e}")
        raise e

    return DataFrame(jdf, self.sparkSession)


def sem_batch_join_jdf_impl(self, right_jdf, prompt, true_criteria, false_criteria, url, model, batch_size, max_tokens, timeout, batch_prompt_size, verbose):
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAIBatchJoin(
        self._jdf, right_jdf, prompt, true_criteria, false_criteria, url, model,
        batch_size, max_tokens, timeout, batch_prompt_size, verbose
    )


def sem_agg_jdf_impl(self, prompt, model, url, groupingColumn, enableColumnSorting, enableRowSorting, max_tokens, model_token_limit):
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAIAgg(self._jdf, prompt, model, url,
                                                                              groupingColumn, enableColumnSorting, enableRowSorting, 
                                                                              max_tokens, model_token_limit)

def sem_groupby_jdf_impl(self, prompt, proxy_model, proxy_url, main_model, main_url, embedder_model, enableColReorder, enableRowReorder, num_clusters, max_tokens):
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAIGroupBy(self._jdf, prompt, proxy_model, proxy_url, main_model, main_url,
                                                                                  embedder_model, enableColReorder, enableRowReorder, num_clusters,
                                                                                  max_tokens)


def sem_embed_jdf_impl(self, prompt,
                       col_out,
                       normalization,
                       batch_size,
                       embedding_model,
                       device,
                       gpu_id,
                       verbose):
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAIEmbed(self._jdf, prompt,
                                                                                col_out,
                                                                                normalization,
                                                                                batch_size,
                                                                                embedding_model,
                                                                                device,
                                                                                gpu_id,
                                                                                verbose)


def sem_sim_join_jdf_impl(self,
                          _jdf,
                          prompt,
                          col_out,
                          batch_size,
                          embedding_model,
                          verbose,
                          calibrate,
                          device,
                          gpu_id,
                          metric_type_str,
                          k,
                          left_embedding_col,
                          right_embedding_col):
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAISimJoin(self._jdf, _jdf,
                                                                                  prompt,
                                                                                  col_out,
                                                                                  batch_size,
                                                                                  embedding_model,
                                                                                  verbose,
                                                                                  calibrate,
                                                                                  device,
                                                                                  gpu_id,
                                                                                  metric_type_str,
                                                                                  k,
                                                                                  left_embedding_col,
                                                                                  right_embedding_col)


def sem_sim_filter_jdf_impl(self,
                            prompt,
                            query,
                            threshold,
                            col_out,
                            embedding_model,
                            embedding_url,
                            timeout,
                            verbose,
                            return_distance):
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAISimFilter(
        self._jdf,
        prompt,
        query,
        threshold,
        col_out,
        embedding_model,
        embedding_url,
        timeout,
        verbose,
        return_distance,
    )


def sem_join_jdf_impl(self,
                      _jdf,
                      prompt,
                      true_criteria,
                      false_criteria,
                      prompt_prefix,
                      prompt_suffix,
                      lower,
                      append_right,
                      do_reorder_columns,
                      do_reorder_rows,
                      url,
                      batch_size,
                      model,
                      embedding_model,
                      max_tokens,
                      temperature,
                      optimized,
                      verbose,
                      stop,
                      timeout,
                      calibrate,
                      device,
                      gpu_id,
                      recall_target,
                      failure_probability,
                      precision_target,
                      sim_join_only,
                      map_prompt,
                      distributed,
                      left_embedding_col,
                      right_embedding_col,
                      confidence_level,
                      error_tolerance,
                      is_approximate,
                      max_concurrent_requests=256,
                      proxy_keep_ratio=0.25,
                      proxy_sample=False,
                      sampling_method="random",
                      sample_size=500,
                      imbalance_strategy="weighted",
                      confidence_threshold=0.5,
                      proxy_model_path="",
                      proxy_warm_start=False,
                ):
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAIJoin(self._jdf, _jdf,
                                                                               prompt,
                                                                               true_criteria,
                                                                               false_criteria,
                                                                               prompt_prefix,
                                                                               prompt_suffix,
                                                                               lower,
                                                                               append_right,
                                                                               do_reorder_columns,
                                                                               do_reorder_rows,
                                                                               url,
                                                                               batch_size,
                                                                               model,
                                                                               embedding_model,
                                                                               max_tokens,
                                                                               temperature,
                                                                               optimized,
                                                                               verbose,
                                                                               stop,
                                                                               timeout,
                                                                               calibrate,
                                                                               device,
                                                                               gpu_id,
                                                                               recall_target,
                                                                               failure_probability,
                                                                               precision_target,
                                                                               sim_join_only,
                                                                               map_prompt,
                                                                               distributed,
                                                                               left_embedding_col,
                                                                               right_embedding_col,
                                                                               confidence_level,
                                                                               error_tolerance,
                                                                               is_approximate,
                                                                               max_concurrent_requests,
                                                                               float(proxy_keep_ratio),
                                                                               bool(proxy_sample),
                                                                               sampling_method,
                                                                               int(sample_size),
                                                                               imbalance_strategy,
                                                                               float(confidence_threshold),
                                                                               proxy_model_path,
                                                                               bool(proxy_warm_start))


def sem_map_jdf_impl(self, prompt,
                     col_out,
                     append_right,
                     do_reorder_columns,
                     do_reorder_rows,
                     url,
                     batch_size,
                     model,
                     max_tokens,
                     temperature,
                     verbose,
                     stop,
                     timeout):
    
    # if image_embeddings is not None:
    #     # Convert Python list to Java String array
    #     gateway = self._session._jvm._gateway_client.gateway
    #     java_embeddings = gateway.new_array(self._session._jvm.java.lang.String, len(image_embeddings))
    #     for i, emb in enumerate(image_embeddings):
    #         java_embeddings[i] = emb
    # else:
    #     java_embeddings = None
    
    return self._session._jvm.org.apache.spark.sql.SemanticDataFrame.semAIMap(self._jdf, prompt,
                                                                              col_out,
                                                                              append_right,
                                                                              do_reorder_columns,
                                                                              do_reorder_rows,
                                                                              url,
                                                                              batch_size,
                                                                              model,
                                                                              max_tokens,
                                                                              temperature,
                                                                              verbose,
                                                                              stop,
                                                                              timeout)




def register_semantic_operators():
    print("Registering some method for scala implicit conversion, also no need to invoke scala method with parameter _jdf...")

    print("Registering sem_filter_jdf...")
    DataFrame.sem_filter_jdf = sem_filter_jdf_impl
    print("Registering sem_filter_cascaded_jdf...")
    DataFrame.sem_filter_cascaded_jdf = sem_filter_cascaded_jdf_impl
    print("Registering sem_approximate_filter_jdf...")
    DataFrame.sem_approximate_filter_jdf = sem_approximate_filter_jdf_impl
    print("Registering sem_batch_filter_jdf...")
    DataFrame.sem_batch_filter_jdf = sem_batch_filter_jdf_impl
    print("Registering sem_approximate_join_jdf...")
    DataFrame.sem_approximate_join_jdf = sem_approximate_join_jdf_impl
    print("Registering sem_batch_join_jdf...")
    DataFrame.sem_batch_join_jdf = sem_batch_join_jdf_impl
    print("Registering sem_agg_jdf...")
    DataFrame.sem_agg_jdf = sem_agg_jdf_impl
    print("Registering sem_groupby_jdf...")
    DataFrame.sem_groupby_jdf = sem_groupby_jdf_impl
    print("Registering sem_embed_jdf...")
    DataFrame.sem_embed_jdf = sem_embed_jdf_impl
    print("Registering sem_sim_join_jdf...")
    DataFrame.sem_sim_join_jdf = sem_sim_join_jdf_impl
    print("Registering sem_sim_filter_jdf...")
    DataFrame.sem_sim_filter_jdf = sem_sim_filter_jdf_impl
    print("Registering sem_join_jdf...")
    DataFrame.sem_join_jdf = sem_join_jdf_impl
    print("Registering sem_map_jdf...")
    DataFrame.sem_map_jdf = sem_map_jdf_impl


    print("Registering sem_topk...")
    DataFrame.sem_topk = sem_topk_impl
    print("Registering sem_map...")
    DataFrame.sem_map = sem_map_impl
    print("Registering sem_join...")
    DataFrame.sem_join = sem_join_impl
    print("Registering sem_sim_join...")
    DataFrame.sem_sim_join = sem_sim_join_impl
    print("Registering sem_sim_filter...")
    DataFrame.sem_sim_filter = sem_sim_filter_impl
    print("Registering sem_groupby...")
    DataFrame.sem_groupby = sem_groupby_impl
    print("Registering sem_agg...")
    DataFrame.sem_agg = sem_agg_impl
    print("Registering sem_filter...")
    DataFrame.sem_filter = sem_filter_impl
    print("Registering sem_embed...")
    DataFrame.sem_embed = sem_embed_impl
    print("Registering sem_approximate_filter...")
    DataFrame.sem_approximate_filter = sem_approximate_filter_impl
    print("Registering sem_batch_filter...")
    DataFrame.sem_batch_filter = sem_batch_filter_impl
    print("Registering sem_approximate_join...")
    DataFrame.sem_approximate_join = sem_approximate_join_impl
    print("Registering sem_batch_join...")
    DataFrame.sem_batch_join = sem_batch_join_impl

register_semantic_operators()
