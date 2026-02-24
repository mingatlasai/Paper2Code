"""Factory registry for config-driven component construction.

This module centralizes name-to-class resolution and stage-specific object
assembly for the THREADS reproduction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Type


class RegistryError(Exception):
    """Base exception for registry failures."""


class RegistryLookupError(RegistryError):
    """Raised when a registry key cannot be resolved."""


class RegistryConstructionError(RegistryError):
    """Raised when component construction fails."""


class RegistryInvariantError(RegistryConstructionError):
    """Raised when paper/config invariants are violated."""


_KIND_PATCH_ENCODER = "patch_encoder"
_KIND_SLIDE_ENCODER = "slide_encoder"
_KIND_RNA_ENCODER = "rna_encoder"
_KIND_DNA_ENCODER = "dna_encoder"
_KIND_LOSS = "loss"
_KIND_MODEL = "model"
_KIND_TRAIN_MODULE = "train_module"
_KIND_CALLBACK = "callback"
_KIND_EVALUATOR = "evaluator"


@dataclass(frozen=True)
class _Target:
    """Lazy import target for a registered class."""

    module_path: str
    class_name: str


class Registry:
    """Config-driven component registry with stage-aware builders."""

    def __init__(self) -> None:
        self._targets: Dict[str, Dict[str, _Target]] = {
            _KIND_PATCH_ENCODER: {
                "conchv1.5": _Target("src.models.patch_encoder", "ConchPatchEncoder"),
            },
            _KIND_SLIDE_ENCODER: {
                "threads_abmil_gated": _Target(
                    "src.models.slide_encoder_threads",
                    "ThreadsSlideEncoder",
                ),
            },
            _KIND_RNA_ENCODER: {
                "scgpt": _Target("src.models.rna_encoder_scgpt", "ScGPTRNAEncoder"),
            },
            _KIND_DNA_ENCODER: {
                "dna_mlp": _Target("src.models.dna_encoder_mlp", "DNAMLPEncoder"),
            },
            _KIND_LOSS: {
                "infonce": _Target("src.models.losses", "ContrastiveLoss"),
            },
            _KIND_MODEL: {
                "threads": _Target("src.models.threads_model", "ThreadsModel"),
            },
            _KIND_TRAIN_MODULE: {
                "pretrain_module": _Target("src.train.pretrain_module", "PretrainModule"),
                "finetune_module": _Target("src.train.finetune_module", "FinetuneModule"),
            },
            _KIND_CALLBACK: {
                "rankme": _Target("src.train.callbacks_rankme", "RankMeCallback"),
            },
            _KIND_EVALUATOR: {
                "embedding_exporter": _Target("src.eval.embedding_export", "EmbeddingExporter"),
                "linear_probe": _Target("src.eval.linear_probe", "LinearProbeEvaluator"),
                "survival_coxnet": _Target("src.eval.survival_eval", "SurvivalEvaluator"),
                "retrieval_l2": _Target("src.eval.retrieval_eval", "RetrievalEvaluator"),
                "prompting_molecular": _Target(
                    "src.eval.prompting_eval",
                    "PromptingEvaluator",
                ),
                "stats_analyzer": _Target("src.eval.stats_tests", "StatsAnalyzer"),
            },
        }

    def resolve(self, kind: str, name: str) -> Type[Any]:
        """Resolve a registered class by kind and key."""
        normalized_kind: str = _normalize_key(kind)
        normalized_name: str = _normalize_key(name)

        kind_map: Optional[Dict[str, _Target]] = self._targets.get(normalized_kind)
        if kind_map is None:
            raise RegistryLookupError(
                f"Unknown registry kind: {kind!r}. Available kinds: {sorted(self._targets.keys())}."
            )

        target: Optional[_Target] = kind_map.get(normalized_name)
        if target is None:
            raise RegistryLookupError(
                f"Unknown registry key {name!r} for kind {kind!r}. "
                f"Available: {sorted(kind_map.keys())}."
            )
        return _import_class(target)

    def build_patch_encoder(self, cfg: Any) -> Any:
        """Build patch encoder."""
        cfg_dict: Dict[str, Any] = _to_dict(cfg)

        model_name: str = _as_str(
            _first_present(
                cfg_dict,
                (
                    ("model", "patch_encoder", "name"),
                    ("model_threads", "model", "patch_encoder", "name"),
                    ("model_threads", "threads", "patch_encoder", "name"),
                ),
                default="CONCHV1.5",
            ),
            default="CONCHV1.5",
        )
        normalized_model_name: str = _normalize_model_alias(model_name)
        if normalized_model_name != "conchv1.5":
            raise RegistryInvariantError(
                f"Only CONCHV1.5 is supported for THREADS patch encoding, got {model_name!r}."
            )

        input_resize: int = _as_int(
            _first_present(
                cfg_dict,
                (
                    ("preprocessing", "patch_encoder_input_resize"),
                    ("model_threads", "model", "patch_encoder", "input_resize"),
                ),
                default=448,
            ),
            default=448,
        )
        if input_resize != 448:
            raise RegistryInvariantError(
                f"Patch encoder resize must be 448, got {input_resize!r}."
            )

        device: str = _as_str(
            _first_present(
                cfg_dict,
                (
                    ("runtime", "device"),
                    ("runtime", "accelerator"),
                    ("embedding_extraction", "hardware", "device"),
                ),
                default="cpu",
            ),
            default="cpu",
        )
        precision: str = _as_str(
            _first_present(
                cfg_dict,
                (
                    ("embedding_extraction", "hardware", "precision"),
                    ("pretrain_public", "pretrain_public", "pretraining_runtime_constraints", "hardware", "precision"),
                    ("train_pretrain", "pretrain", "precision", "mode"),
                ),
                default="fp32",
            ),
            default="fp32",
        )

        patch_encoder_class: Type[Any] = self.resolve(_KIND_PATCH_ENCODER, normalized_model_name)
        return patch_encoder_class(model_name=model_name, device=device, precision=precision)

    def build_threads_model(self, cfg: Any) -> Any:
        """Build THREADS multimodal model with all branches and loss."""
        cfg_dict: Dict[str, Any] = _to_dict(cfg)

        slide_encoder: Any = self._build_slide_encoder(cfg_dict)
        rna_encoder: Any = self._build_rna_encoder(cfg_dict)
        dna_encoder: Any = self._build_dna_encoder(cfg_dict)
        patch_encoder: Any = self.build_patch_encoder(cfg_dict)
        loss_fn: Any = self._build_contrastive_loss(cfg_dict)

        model_class: Type[Any] = self.resolve(_KIND_MODEL, "threads")
        return model_class(
            patch_encoder=patch_encoder,
            slide_encoder=slide_encoder,
            rna_encoder=rna_encoder,
            dna_encoder=dna_encoder,
            loss_fn=loss_fn,
        )

    def build_pretrain_module(self, cfg: Any, model: Any) -> Any:
        """Build pretraining module."""
        cfg_dict: Dict[str, Any] = _to_dict(cfg)
        pretrain_cfg: Dict[str, Any] = _deep_get_dict(cfg_dict, ("train_pretrain", "pretrain"), {})
        if not pretrain_cfg:
            raise RegistryConstructionError(
                "Missing train_pretrain.pretrain config for PretrainModule construction."
            )

        optim_cfg: Dict[str, Any] = dict(_deep_get_dict(pretrain_cfg, ("optimizer",), {}))
        sched_cfg: Dict[str, Any] = dict(_deep_get_dict(pretrain_cfg, ("scheduler",), {}))
        module_class: Type[Any] = self.resolve(_KIND_TRAIN_MODULE, "pretrain_module")
        return module_class(model=model, optim_cfg=optim_cfg, sched_cfg=sched_cfg)

    def build_finetune_module(self, cfg: Any, slide_encoder: Any, task_type: str) -> Any:
        """Build finetuning module."""
        cfg_dict: Dict[str, Any] = _to_dict(cfg)

        finetune_cfg: Dict[str, Any] = _deep_get_dict(cfg_dict, ("train_finetune", "finetune"), {})
        if not finetune_cfg:
            raise RegistryConstructionError(
                "Missing train_finetune.finetune config for FinetuneModule construction."
            )

        contracts_cfg: Dict[str, Any] = _deep_get_dict(finetune_cfg, ("contracts",), {})
        head_out_dim: int = _as_int(
            _first_present(
                contracts_cfg,
                (("slide_embedding_dim",),),
                default=_first_present(
                    cfg_dict,
                    (
                        ("model", "slide_embedding_dim"),
                        ("model_threads", "model", "slide_embedding_dim"),
                    ),
                    default=1024,
                ),
            ),
            default=1024,
        )

        recipe_name: str = _as_str(
            _first_present(
                finetune_cfg,
                (("selector", "default_recipe"),),
                default="threads",
            ),
            default="threads",
        )
        recipe_cfg: Dict[str, Any] = _deep_get_dict(finetune_cfg, (recipe_name,), {})
        if not recipe_cfg:
            recipe_cfg = _deep_get_dict(finetune_cfg, ("threads",), {})

        optim_cfg: Dict[str, Any] = dict(_deep_get_dict(recipe_cfg, ("optimizer",), {}))
        module_class: Type[Any] = self.resolve(_KIND_TRAIN_MODULE, "finetune_module")
        return module_class(
            slide_encoder=slide_encoder,
            head_out_dim=head_out_dim,
            task_type=task_type,
            optim_cfg=optim_cfg,
        )

    def build_linear_probe_evaluator(self, cfg: Any) -> Any:
        """Build linear-probe evaluator with fixed paper defaults."""
        cfg_dict: Dict[str, Any] = _to_dict(cfg)
        linear_cfg: Dict[str, Any] = _deep_get_dict(cfg_dict, ("linear_probe", "classification"), {})

        c_value: float = _as_float(_first_present(linear_cfg, (("C",),), default=0.5), default=0.5)
        max_iter: int = _as_int(_first_present(linear_cfg, (("max_iter",),), default=10000), default=10000)
        solver: str = _as_str(_first_present(linear_cfg, (("solver",),), default="lbfgs"), default="lbfgs")
        class_weight: str = _as_str(
            _first_present(linear_cfg, (("class_weight",),), default="balanced"),
            default="balanced",
        )

        if c_value != 0.5 or max_iter != 10000 or solver != "lbfgs" or class_weight != "balanced":
            raise RegistryInvariantError(
                "Linear probe settings must match paper defaults: "
                "C=0.5, solver=lbfgs, max_iter=10000, class_weight=balanced."
            )

        evaluator_class: Type[Any] = self.resolve(_KIND_EVALUATOR, "linear_probe")
        return evaluator_class(
            c_value=c_value,
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
        )

    def build_survival_evaluator(self, cfg: Any, task_name: str, model_name: str) -> Any:
        """Build survival evaluator with alpha resolution policy."""
        cfg_dict: Dict[str, Any] = _to_dict(cfg)
        survival_cfg: Dict[str, Any] = _deep_get_dict(cfg_dict, ("linear_probe", "survival"), {})

        max_iter: int = _as_int(
            _first_present(survival_cfg, (("max_iter",),), default=10000),
            default=10000,
        )
        if max_iter != 10000:
            raise RegistryInvariantError("Survival max_iter must be 10000.")

        alpha_defaults: Dict[str, Any] = _deep_get_dict(survival_cfg, ("alpha",), {})
        alpha_os: float = _as_float(
            _first_present(alpha_defaults, (("overall_survival",),), default=0.07),
            default=0.07,
        )
        alpha_pfs: float = _as_float(
            _first_present(alpha_defaults, (("progression_free_survival",),), default=0.01),
            default=0.01,
        )
        alpha_overrides: Iterable[Mapping[str, Any]] = _as_sequence_of_mapping(
            _first_present(survival_cfg, (("alpha_exceptions",),), default=[]),
        )

        resolved_alpha: float = _resolve_survival_alpha(
            task_name=task_name,
            model_name=model_name,
            alpha_os=alpha_os,
            alpha_pfs=alpha_pfs,
            overrides=alpha_overrides,
        )

        evaluator_class: Type[Any] = self.resolve(_KIND_EVALUATOR, "survival_coxnet")
        return evaluator_class(alpha=resolved_alpha, max_iter=max_iter)

    def build_retrieval_evaluator(self, cfg: Any) -> Any:
        """Build retrieval evaluator."""
        _ = _to_dict(cfg)
        evaluator_class: Type[Any] = self.resolve(_KIND_EVALUATOR, "retrieval_l2")
        return evaluator_class(metric="l2", top_k=[1, 5, 10])

    def build_prompting_evaluator(self, cfg: Any) -> Any:
        """Build prompting evaluator."""
        _ = _to_dict(cfg)
        evaluator_class: Type[Any] = self.resolve(_KIND_EVALUATOR, "prompting_molecular")
        return evaluator_class()

    def build_stats_analyzer(self, cfg: Any) -> Any:
        """Build statistical analyzer."""
        _ = _to_dict(cfg)
        analyzer_class: Type[Any] = self.resolve(_KIND_EVALUATOR, "stats_analyzer")
        return analyzer_class()

    def build_preprocess_components(self, cfg: Any) -> Dict[str, Any]:
        """Build preprocess-stage components."""
        cfg_dict: Dict[str, Any] = _to_dict(cfg)
        return {
            "patch_encoder": self.build_patch_encoder(cfg_dict),
        }

    def build_pretrain_components(self, cfg: Any) -> Dict[str, Any]:
        """Build pretrain-stage components."""
        cfg_dict: Dict[str, Any] = _to_dict(cfg)
        model: Any = self.build_threads_model(cfg_dict)
        module: Any = self.build_pretrain_module(cfg_dict, model=model)
        callback_class: Type[Any] = self.resolve(_KIND_CALLBACK, "rankme")

        rankme_eps: float = _as_float(
            _first_present(
                cfg_dict,
                (
                    ("pretraining", "training", "rankme_eps"),
                    ("train_pretrain", "pretrain", "rankme", "eps"),
                ),
                default=1.0e-7,
            ),
            default=1.0e-7,
        )
        start_epoch: int = _as_int(
            _first_present(
                cfg_dict,
                (
                    ("train_pretrain", "pretrain", "rankme", "first_monitor_epoch"),
                    ("pretraining", "scheduler", "warmup_epochs"),
                ),
                default=5,
            ),
            default=5,
        )
        callback: Any = callback_class(start_epoch=start_epoch, eps=rankme_eps)
        return {
            "model": model,
            "pretrain_module": module,
            "rankme_callback": callback,
        }

    def build_embed_components(self, cfg: Any) -> Dict[str, Any]:
        """Build embed-stage components."""
        cfg_dict: Dict[str, Any] = _to_dict(cfg)
        evaluator_class: Type[Any] = self.resolve(_KIND_EVALUATOR, "embedding_exporter")

        feature_store_cls: Type[Any] = _import_class(_Target("src.data.feature_store", "FeatureStore"))
        feature_root: str = _as_str(
            _first_present(
                cfg_dict,
                (
                    ("pretrain_public", "pretrain_public", "io_roots", "features_root"),
                    ("downstream_public", "downstream_public", "io", "embeddings_root"),
                ),
                default="data/processed/features",
            ),
            default="data/processed/features",
        )
        feature_fmt: str = _as_str(
            _first_present(
                cfg_dict,
                (
                    ("pretrain_public", "pretrain_public", "manifests", "format"),
                ),
                default="hdf5",
            ),
            default="hdf5",
        )
        feature_store: Any = feature_store_cls(root_dir=feature_root, fmt=feature_fmt)

        device: str = _as_str(
            _first_present(
                cfg_dict,
                (
                    ("runtime", "device"),
                    ("runtime", "accelerator"),
                ),
                default="cpu",
            ),
            default="cpu",
        )
        model_ckpt: str = _as_str(
            _first_present(
                cfg_dict,
                (
                    ("runtime", "model_ckpt"),
                    ("train_pretrain", "pretrain", "checkpointing", "resume_from"),
                ),
                default="",
            ),
            default="",
        )
        return {
            "embedding_exporter": evaluator_class(
                feature_store=feature_store,
                model_ckpt=model_ckpt,
                device=device,
            ),
        }

    def build_eval_components(
        self,
        cfg: Any,
        task_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build eval-stage components."""
        cfg_dict: Dict[str, Any] = _to_dict(cfg)
        context: Mapping[str, Any] = task_context or {}
        task_name: str = _as_str(context.get("task_name", ""), default="")
        model_name: str = _as_str(context.get("model_name", "THREADS"), default="THREADS")

        return {
            "linear_probe": self.build_linear_probe_evaluator(cfg_dict),
            "survival": self.build_survival_evaluator(
                cfg_dict,
                task_name=task_name,
                model_name=model_name,
            ),
            "retrieval": self.build_retrieval_evaluator(cfg_dict),
            "prompting": self.build_prompting_evaluator(cfg_dict),
            "stats": self.build_stats_analyzer(cfg_dict),
        }

    def _build_slide_encoder(self, cfg_dict: Mapping[str, Any]) -> Any:
        """Build slide encoder."""
        slide_cfg: Dict[str, Any] = _merge_dicts(
            _deep_get_dict(cfg_dict, ("model", "slide_encoder"), {}),
            _deep_get_dict(cfg_dict, ("model_threads", "model", "slide_encoder"), {}),
            _deep_get_dict(cfg_dict, ("model_threads", "threads", "slide_encoder"), {}),
        )
        encoder_type: str = _as_str(
            _first_present(slide_cfg, (("type",),), default="ABMIL gated attention"),
            default="ABMIL gated attention",
        )
        if encoder_type != "ABMIL gated attention":
            raise RegistryInvariantError(
                f"Slide encoder type must be 'ABMIL gated attention', got {encoder_type!r}."
            )

        in_dim: int = _as_int(_first_present(slide_cfg, (("in_dim",),), default=768), default=768)
        hidden_dim: int = _as_int(
            _first_present(slide_cfg, (("hidden_dim",),), default=1024),
            default=1024,
        )
        out_dim: int = _as_int(_first_present(slide_cfg, (("out_dim",),), default=1024), default=1024)
        n_heads: int = _as_int(
            _first_present(slide_cfg, (("attention_heads_main",), ("n_heads",)), default=2),
            default=2,
        )
        dropout: float = _as_float(
            _first_present(
                slide_cfg,
                (
                    ("pre_attention", "dropout"),
                    ("pre_attention_dropout",),
                ),
                default=0.1,
            ),
            default=0.1,
        )
        if out_dim != 1024:
            raise RegistryInvariantError(f"THREADS slide encoder out_dim must be 1024, got {out_dim}.")

        slide_encoder_class: Type[Any] = self.resolve(_KIND_SLIDE_ENCODER, "threads_abmil_gated")
        return slide_encoder_class(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

    def _build_rna_encoder(self, cfg_dict: Mapping[str, Any]) -> Any:
        """Build RNA encoder."""
        rna_cfg: Dict[str, Any] = _merge_dicts(
            _deep_get_dict(cfg_dict, ("model", "rna_encoder"), {}),
            _deep_get_dict(cfg_dict, ("model_threads", "model", "rna_encoder"), {}),
            _deep_get_dict(cfg_dict, ("model_threads", "threads", "rna_encoder"), {}),
        )
        out_dim: int = _as_int(
            _first_present(
                rna_cfg,
                (
                    ("projection_out_dim",),
                ),
                default=1024,
            ),
            default=1024,
        )
        trainable: bool = _as_bool(_first_present(rna_cfg, (("trainable",),), default=True), default=True)
        ckpt_path: str = _as_str(
            _first_present(rna_cfg, (("checkpoint_path",),), default=""),
            default="",
        )
        if out_dim != 1024:
            raise RegistryInvariantError(f"RNA encoder projection_out_dim must be 1024, got {out_dim}.")

        rna_encoder_class: Type[Any] = self.resolve(_KIND_RNA_ENCODER, "scgpt")
        return rna_encoder_class(ckpt_path=ckpt_path, out_dim=out_dim, trainable=trainable)

    def _build_dna_encoder(self, cfg_dict: Mapping[str, Any]) -> Any:
        """Build DNA encoder."""
        dna_cfg: Dict[str, Any] = _merge_dicts(
            _deep_get_dict(cfg_dict, ("model", "dna_encoder"), {}),
            _deep_get_dict(cfg_dict, ("model_threads", "model", "dna_encoder"), {}),
            _deep_get_dict(cfg_dict, ("model_threads", "threads", "dna_encoder"), {}),
        )
        in_dim: int = _as_int(_first_present(dna_cfg, (("input_dim",),), default=1673), default=1673)
        hidden_dim: int = _as_int(_first_present(dna_cfg, (("hidden_dim",),), default=in_dim), default=in_dim)
        out_dim: int = _as_int(_first_present(dna_cfg, (("output_dim",),), default=1024), default=1024)
        dropout: float = _as_float(_first_present(dna_cfg, (("dropout",),), default=0.2), default=0.2)

        if in_dim != 1673 or out_dim != 1024:
            raise RegistryInvariantError(
                f"DNA encoder dims must be 1673->1024, got {in_dim}->{out_dim}."
            )
        dna_encoder_class: Type[Any] = self.resolve(_KIND_DNA_ENCODER, "dna_mlp")
        return dna_encoder_class(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

    def _build_contrastive_loss(self, cfg_dict: Mapping[str, Any]) -> Any:
        """Build InfoNCE-style loss."""
        objective_cfg: Dict[str, Any] = _merge_dicts(
            _deep_get_dict(cfg_dict, ("model", "objective"), {}),
            _deep_get_dict(cfg_dict, ("model_threads", "model", "objective"), {}),
            _deep_get_dict(cfg_dict, ("model_threads", "threads", "multimodal_objective"), {}),
        )
        loss_name: str = _as_str(
            _first_present(objective_cfg, (("loss_name",), ("type",)), default="infonce"),
            default="infonce",
        )
        normalized_loss_name: str = _normalize_model_alias(loss_name)
        if "infonce" not in normalized_loss_name and "contrastive" not in normalized_loss_name:
            raise RegistryInvariantError(
                f"Contrastive loss must be InfoNCE-style, got {loss_name!r}."
            )

        temperature_raw: Any = _first_present(
            objective_cfg,
            (("temperature",),),
            default=None,
        )
        if temperature_raw is None:
            raise RegistryConstructionError(
                "Missing objective temperature. Provide explicit config at "
                "model.objective.temperature (or equivalent model_threads path)."
            )
        temperature: float = _as_float(temperature_raw, default=0.07)
        bidirectional: bool = _as_bool(
            _first_present(objective_cfg, (("bidirectional",),), default=True),
            default=True,
        )

        loss_class: Type[Any] = self.resolve(_KIND_LOSS, "infonce")
        return loss_class(temperature=temperature, bidirectional=bidirectional)


def get_default_registry() -> Registry:
    """Return a default registry instance."""
    return Registry()


def _import_class(target: _Target) -> Type[Any]:
    """Import and return a class from a lazy target."""
    try:
        module: Any = importlib.import_module(target.module_path)
    except Exception as exc:  # noqa: BLE001
        raise RegistryLookupError(
            f"Failed to import module {target.module_path!r}."
        ) from exc

    if not hasattr(module, target.class_name):
        raise RegistryLookupError(
            f"Module {target.module_path!r} does not define class {target.class_name!r}."
        )
    klass: Any = getattr(module, target.class_name)
    if not isinstance(klass, type):
        raise RegistryLookupError(
            f"Resolved symbol {target.class_name!r} from {target.module_path!r} is not a class."
        )
    return klass


def _normalize_key(value: str) -> str:
    """Normalize registry kind/key."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_model_alias(value: str) -> str:
    """Normalize model aliases."""
    normalized: str = _normalize_key(value)
    alias_map: Dict[str, str] = {
        "conchv15": "conchv1.5",
        "conch_v1.5": "conchv1.5",
        "conch_v15": "conchv1.5",
        "conch": "conchv1.5",
        "cross_modal_contrastive_learning_(infonce_style)": "infonce",
        "cross_modal_contrastive_learning_(infonce_style)": "infonce",
    }
    return alias_map.get(normalized, normalized)


def _to_dict(cfg: Any) -> Dict[str, Any]:
    """Convert supported config objects into a plain dictionary."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
        result: Any = cfg.to_dict()
        if isinstance(result, dict):
            return dict(result)
    try:
        from omegaconf import OmegaConf  # Local import to avoid hard dependency at import-time.

        if OmegaConf.is_config(cfg):
            container: Any = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(container, dict):
                return dict(container)
    except Exception:
        pass
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise RegistryConstructionError(
        f"Unsupported config object type: {type(cfg).__name__}."
    )


def _deep_get(data: Mapping[str, Any], path: Tuple[str, ...], default: Any) -> Any:
    """Get nested value from mapping using a tuple path."""
    cursor: Any = data
    for key in path:
        if not isinstance(cursor, Mapping):
            return default
        if key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _deep_get_dict(data: Mapping[str, Any], path: Tuple[str, ...], default: Mapping[str, Any]) -> Dict[str, Any]:
    """Get nested mapping and coerce to dict."""
    result: Any = _deep_get(data, path, default)
    if isinstance(result, Mapping):
        return dict(result)
    return dict(default)


def _first_present(
    data: Mapping[str, Any],
    paths: Iterable[Tuple[str, ...]],
    default: Any,
) -> Any:
    """Return the first non-None nested value across candidate paths."""
    for path in paths:
        value: Any = _deep_get(data, path, None)
        if value is not None:
            return value
    return default


def _merge_dicts(*dicts: Mapping[str, Any]) -> Dict[str, Any]:
    """Shallow merge dictionaries from left to right."""
    merged: Dict[str, Any] = {}
    for item in dicts:
        if isinstance(item, Mapping):
            merged.update(dict(item))
    return merged


def _as_str(value: Any, default: str) -> str:
    """Coerce value to str with explicit default."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _as_int(value: Any, default: int) -> int:
    """Coerce value to int with explicit default."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise RegistryConstructionError(f"Expected int, got bool: {value!r}.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise RegistryConstructionError(f"Expected integer-like float, got {value!r}.")
        return int(value)
    if isinstance(value, str):
        stripped: str = value.strip()
        if stripped == "":
            return default
        try:
            return int(stripped)
        except ValueError as exc:
            raise RegistryConstructionError(f"Expected int, got {value!r}.") from exc
    raise RegistryConstructionError(f"Expected int, got {type(value).__name__}.")


def _as_float(value: Any, default: float) -> float:
    """Coerce value to float with explicit default."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise RegistryConstructionError(f"Expected float, got bool: {value!r}.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped: str = value.strip()
        if stripped == "":
            return default
        try:
            return float(stripped)
        except ValueError as exc:
            raise RegistryConstructionError(f"Expected float, got {value!r}.") from exc
    raise RegistryConstructionError(f"Expected float, got {type(value).__name__}.")


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce value to bool with explicit default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized: str = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise RegistryConstructionError(f"Expected bool, got {type(value).__name__}: {value!r}.")


def _as_sequence_of_mapping(value: Any) -> Iterable[Mapping[str, Any]]:
    """Validate a sequence of mappings."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise RegistryConstructionError("Expected a list/tuple of mappings.")
    for item in value:
        if not isinstance(item, Mapping):
            raise RegistryConstructionError("Expected mapping item in sequence.")
    return value


def _resolve_survival_alpha(
    task_name: str,
    model_name: str,
    alpha_os: float,
    alpha_pfs: float,
    overrides: Iterable[Mapping[str, Any]],
) -> float:
    """Resolve CoxNet alpha by override and endpoint heuristics."""
    task_norm: str = task_name.strip().lower()
    model_norm: str = model_name.strip().lower()

    for item in overrides:
        ov_task: str = _as_str(item.get("task"), default="").strip().lower()
        ov_model: str = _as_str(item.get("model"), default="").strip().lower()
        if ov_task == task_norm and ov_model == model_norm:
            return _as_float(item.get("alpha"), default=alpha_os)

    pfs_keywords: Tuple[str, ...] = (
        "progression-free survival",
        "progression free survival",
        "pfs",
    )
    for keyword in pfs_keywords:
        if keyword in task_norm:
            return alpha_pfs
    return alpha_os
