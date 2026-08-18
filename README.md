# Induction Heads in Pythia-160m: Detection and Emergence Across Training

A mechanistic interpretability study of EleutherAI's Pythia models. Using raw PyTorch forward
hooks and Hugging Face transformers, I locate the induction circuit in pythia-160m,
**causally test** it by ablation, track when it forms across training using Pythia's
intermediate checkpoints, and check whether that timing holds across model scale (70m–410m).

The core result (induction heads and their emergence as a phase transition) is a reproduction of
known findings; the causal ablation, the head-by-head developmental tracking, and the cross-scale
comparison extend it. It is framed as a rigorous case study, not a claim of novel discovery.

## Background

**Induction** is the ability to continue a repeated pattern: shown … [A][B] … [A], a model
with induction predicts [B]. Mechanistically, this is implemented by induction heads, which are
attention heads that, at the second [A], attend back to the token that followed [A] last
time and copy it forward. Induction is the most reproducible phenomenon in mechanistic
interpretability and is known to underpin a large part of in-context learning, which makes it
an ideal target for a short, reliable study.

## Method

1. **Behavioural check (does the model do it at all?).** I construct sequences of random tokens
   and repeat them ([P][P]), then measure per-position cross-entropy loss. If the model uses
   induction, loss should fall sharply at the repeat boundary because the second half is fully
   predictable from the first.
2. **Internal capture.** I load the model with eager attention (attn_implementation="eager")
   so attention weights are exposed, register PyTorch forward hooks on each attention module to
   demonstrate the hook mechanism, and read out the attention probability tensors
   (output_attentions=True) for analysis.
3. **Per-head induction score.** For every (layer, head) I measure how much attention, when
   querying from the repeated half, lands on the induction target (the token after the previous
   occurrence). High score = induction head. This localizes the behaviour to specific heads.
4. **Emergence across training.** I re-run the behavioural metric on several of Pythia's
   intermediate training checkpoints to see *when* induction appears.

All analysis is plain PyTorch tensor work; no interpretability wrapper library is used.

## Results

### The model does induction (strongly)

On repeated random sequences, mean next-token loss was **24.41 on the first occurrence** and
**5.29 on the repeated half** — an induction drop of **19.13**. A large, unambiguous effect,
confirming the behaviour is present before any internal analysis.

### Induction localizes to a few mid-layer heads

The highest-scoring heads (attention score in brackets, max 1.0):

| Rank | Layer | Head | Induction score |
|-----:|:-----:|:----:|:---------------:|
| 1 | 4 | 6  | 0.98 |
| 2 | 8 | 2  | 0.94 |
| 3 | 4 | 10 | 0.91 |
| 4 | 5 | 0  | 0.79 |
| 5 | 5 | 6  | 0.75 |

Scores near 0.98 mean the head places almost all of its attention on exactly the induction
target. The heads sit in the middle layers (4, 5, 8), consistent with the standard two-stage
induction circuit (an earlier "previous-token" head feeding a later "copy" head). The full
layer × head heatmap shows a small number of bright cells against an otherwise dark grid.

### Emergence across training: a sharp phase transition

Sampling eleven checkpoints from the start of training to the end reveals that induction is
**absent through step 512, then emerges sharply before step 1,000:**

| Training step | Induction drop |
|--------------:|:--------------:|
| 1     |  0.01 |
| 8     |  0.01 |
| 32    | −0.00 |
| 64    | −0.00 |
| 128   |  0.02 |
| 256   | −0.04 |
| 512   |  0.00 |
| 1,000 | **10.30** |
| 8,000 | 13.11 |
| 64,000 | 12.71 |
| 143,000 (final) | 19.13 |

Through step 512 the induction drop is indistinguishable from zero — the model has not yet built
the circuit. Between **step 512 and step 1,000 it snaps on**, jumping from ≈0 to ≈10, then
consolidates and strengthens to ≈19 by the end of training. This is the well-documented induction
phase transition: the behaviour appears abruptly within a narrow training window rather than
accumulating gradually. The dense early sampling brackets that window to between step 512 and
step 1,000.

### The circuit is causally responsible but redundant

To test whether the identified heads cause induction rather than merely correlate with it, I
ablated them (zeroing each head's contribution at the input to the attention output projection,
where heads are still separable) and re-measured the induction drop.

**Ablating all five top heads together collapsed the induction drop by 64% (19.1 → 6.9)** — strong
causal evidence that this set of heads implements the behaviour. **Single-head ablations, by
contrast, had small and inconsistent effects** (reductions ranging from ~1% to ~14%; the
highest-attention head, L4H6, accounted for only ~7%). This pattern is the signature of a
**redundant, distributed circuit**: removing any one head leaves the others able to do the job, and
only knocking out the group together breaks the behaviour. Therefore, the causal claim rests on the
joint ablation, with single-head results reported as evidence of redundancy rather than
localization to any one head.

### The phase transition is mechanistic, not just behavioural

Tracking each top head's induction score *across* training links the behavioural transition to the
circuit forming. The two layer-4 heads (L4H6, L4H10) switch on **exactly at the step 512→1,000
window** — from ≈0.02 to ≈0.91 — coinciding with the behavioural jump. The layer-5 and layer-8
heads (L5H0, L8H2) come online **much later and gradually**, reaching high scores only by the end of
training. So the circuit assembles in **stages**: the layer-4 heads appear at the phase transition
and are sufficient to produce the behaviour, while additional heads are recruited over the rest of
training (plausibly why the drop keeps rising from ≈10 to ≈19 after the transition). The redundancy
seen in the ablation and the staged assembly seen here is the same story from two angles.

### Emergence timing is roughly scale-invariant

Repeating the emergence analysis on **pythia-70m, 160m, and 410m** shows all three sizes transition
in the **same step 512→1,000 window**:

| Step | 70m | 160m | 410m |
|-----:|:---:|:----:|:----:|
| 256  | 0.04 | −0.04 | 0.00 |
| 512  | 0.05 | 0.00  | −0.14 |
| 1,000 | **9.59** | **10.30** | **8.28** |
| 2,000 | 11.47 | 12.63 | 12.43 |
| 8,000 | 11.79 | 13.11 | 12.89 |
| final | 12.03 | 19.13 | 12.53 |

Larger models do **not** acquire induction meaningfully earlier — the onset step is shared across an
order of magnitude of scale. The sizes differ mainly in final plateau height, though that comparison
is confounded by differing loss magnitudes across models, so the robust claim is about timing, not
magnitude.

## Figures

- figures/per_position_loss.png — loss vs. sequence position; the sharp drop at the repeat
  boundary is the behavioural signature of induction.
- figures/induction_heatmap.png — per-head induction scores (layer × head).
- figures/emergence.png — induction drop vs. training step (the phase transition).
- figures/ablation.png — induction drop, baseline vs. ablating the top heads.
- figures/head_trajectories.png — per-head induction score across training (staged assembly).
- figures/scale.png — emergence across 70m / 160m / 410m.

## Limitations

- **Single model family.** Findings are a case study on Pythia (70m–410m), not a general result
  about all transformers.
- **Causal test is coarse.** Ablation establishes the head *group* is necessary, but the
  inconsistent single-head effects mean the analysis does not cleanly attribute the behaviour to
  individual heads; a more careful path-patching analysis would resolve the internal division of
  labour.
- **Checkpoint spacing.** Emergence is bracketed to the step 512–1,000 window; Pythia has no saved
  checkpoints between 512 and 1,000, so the transition cannot be resolved more finely without a
  differently-logged training run.
- **Cross-size magnitudes not comparable.** The scale comparison supports a claim about emergence
  timing, not plateau height, because loss magnitudes differ across model sizes.
- **One stimulus type.** Repeated random tokens probe copy-style induction specifically, not all
  forms of in-context learning.

## Future work

- **Path patching** to resolve the division of labour among the redundant heads and identify the
  previous-token → copy-head structure directly.
- **Per-checkpoint ablation**:  testing not just *when* heads activate but when they become causally
  *necessary*, fusing the developmental and causal analyses.
- Extending to **larger Pythia sizes** and to **non-copy in-context-learning** tasks to test how far
  the staged-assembly picture generalizes.

## Reproducing

Runs on a free Colab or Kaggle T4 in a few minutes (Parts 1–4); the checkpoint sweep adds a few
hundred MB of download per checkpoint.

```bash
pip install --upgrade transformers
```

Then run pythia_induction.ipynb top to bottom. Notes: load the model with
attn_implementation="eager" (the fast SDPA backend does not expose attention weights), and do
not upgrade torch on Kaggle/Colab — the preinstalled version already matches the GPU image, and
upgrading it breaks the torchvision/GPT-NeoX import.

## Stack

PyTorch · Hugging Face transformers · EleutherAI/pythia-160m (+ training checkpoints) ·
NumPy · Matplotlib
