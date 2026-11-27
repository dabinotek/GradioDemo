# Zero-Shot Machine Translation: Models, Failure Modes, and Practical Diagnostics

**Seminar presentation · ACL/EMNLP-inspired**  
**Design:** white canvas, slim maroon header, restrained academic palette  
**Focus:** mBART-50 and NLLB-200 · English → Kikuyu case study

## Slide 01 — Zero-Shot Machine Translation

- **Models, failure modes, and practical diagnostics**
- A seminar on multilingual encoder-decoder Transformers
- Case study: English source, Kikuyu reference, mBART-50 and NLLB-200 outputs
- Presenter: [Name] · [Affiliation] · [Date]

> Visual: Title slide; slim maroon top banner, centered title, subtle navy subtitle.

## Slide 02 — Motivation: translation beyond high-resource pairs

- Machine translation quality is uneven across languages and directions.
- Low-resource communities need usable translation—not just broad language-count claims.
- Multilingual models transfer knowledge across languages, but transfer can fail.
- We ask: what does zero-shot mean, how do these models differ, and how should failures be diagnosed?

> Visual: Language-pair map with high-resource and low-resource directions.

## Slide 03 — Seminar roadmap

- Translation task and the zero-shot setting
- Model families: mBART-50 and NLLB-200
- Transformer encoder-decoder anatomy
- English → Kikuyu case study and failure diagnosis
- Inference pipelines, evaluation, and takeaways

> Visual: Five-stage horizontal agenda with current section highlighted.

## Slide 04 — Task definition and terminology

- **Source:** English question about carbohydrates in dairy cattle diets.
- **Target:** Kikuyu, represented in NLLB with `kik_Latn`.
- **Reference:** a human-produced Kikuyu translation.
- **Translation direction matters:** English → Kikuyu is not interchangeable with Kikuyu → English.
- A translation can be fluent yet wrong, or generated in an unintended language.

> Visual: Source → model → target, with direction labels on the arrows.

## Slide 05 — What “zero-shot” means

- Zero-shot translation evaluates a language pair without direct parallel examples for that exact pair during the relevant training or fine-tuning stage.
- A model may still have learned each language through other pairs or multilingual objectives.
- Shared representations can support transfer, but do not guarantee it.
- Always specify the checkpoint, training regime, pair, direction, and meaning of “unseen.”

> Visual: Two seen language-pair edges and one dashed unseen pair edge.

## Slide 06 — Case-study outputs

- **English source:** “What is the role of carbohydrates in the diet of dairy cattle?”
- **mBART example:** repeated “main” tokens (degeneration).
- **NLLB example:** French output instead of the Kikuyu reference.
- **Human reference:** “Hihi bata wa irio cia hinya mīrīreinī ya ng'ombe ya iria.”
- These are illustrative observed outputs; they do not establish the root cause by themselves.

> Visual: Three aligned cards for mBART, NLLB, and human reference; red labels identify the observed issues.

## Slide 07 — Two multilingual model families

- **mBART:** multilingual denoising pretraining; translation behavior depends on the selected checkpoint and fine-tuning.
- **NLLB:** translation-focused multilingual training designed for broad many-to-many coverage.
- Both use encoder-decoder Transformer architectures.
- The key differences for this seminar are language support, training data/objectives, checkpoint scale, and language-code conventions.

> Visual: Side-by-side model cards linked to shared Transformer backbone.

## Slide 08 — mBART-50 at a glance

- Common checkpoint: `facebook/mbart-large-50-many-to-many-mmt`.
- 50 languages; about 610M parameters; 12 encoder and 12 decoder layers.
- Example configuration: `d_model=1024`, 16 attention heads, FFN dimension 4096.
- Uses mBART language tokens, e.g. English `en_XX`.
- **Kikuyu is not supported by mBART-50.**

> Visual: Checkpoint facts in a compact table; unsupported Kikuyu shown as a clear caution.

## Slide 09 — NLLB-200 at a glance

- NLLB-200 is a family; public checkpoints include distilled 600M, 1.3B, and 3.3B variants.
- Named example: `facebook/nllb-200-3.3B`, approximately 3.3B parameters.
- 24 encoder and 24 decoder layers; `d_model=1024`, 16 heads, FFN dimension 8192.
- NLLB language codes include English `eng_Latn` and Kikuyu `kik_Latn`.
- Language inclusion does not guarantee equal quality across directions.

> Visual: Checkpoint scale with 3.3B highlighted and language-code tags below.

## Slide 10 — Shared encoder-decoder workflow

- **Input:** text plus source-language information is tokenized.
- **Encoder:** builds contextual representations of the source sequence.
- **Decoder:** consumes a target prefix and attends to encoder states.
- **Output head:** maps decoder states to vocabulary probabilities.
- Generation repeats token by token until a stopping condition is met.

> Visual: Input → encoder memory → decoder → vocabulary distribution → next token.

## Slide 11 — Encoder: source to contextual memory

- Token embeddings are combined with positional information.
- Multi-head self-attention lets each source position use context from other positions.
- Residual connections and LayerNorm support stable information flow.
- A position-wise feed-forward network applies two learned projections with an activation.
- Repeated encoder layers produce contextual source states for cross-attention.

> Visual: Vertical compartment with attention, Add & Norm, FC → activation → FC, and residual paths.

## Slide 12 — Decoder: autoregressive generation

- Decoder input is the target prefix shifted right; the target-language token conditions output language.
- **Masked self-attention** prevents access to future target tokens.
- **Cross-attention** uses decoder queries and encoder memory as keys and values.
- A feed-forward sublayer transforms each position, then a vocabulary projection predicts the next token.
- In inference, generated tokens are appended and decoded iteratively.

> Visual: Masked self-attention → cross-attention + encoder-memory arrow → FFN → token distribution.

## Slide 13 — mBART-50 architecture: checkpoint view

- 12 encoder layers and 12 decoder layers.
- 16 attention heads; model width 1024; FFN width 4096.
- Representative FFN: 1024 → 4096 → GELU → 1024.
- English source ID uses `en_XX`; target ID must be chosen from supported mBART-50 codes.
- This architecture cannot make Kikuyu a supported target language.

> Visual: Grouped encoder/decoder compartments with 12× repeat badges and a memory bridge.

## Slide 14 — NLLB-200 3.3B architecture: checkpoint view

- 24 encoder layers and 24 decoder layers.
- 16 attention heads; model width 1024; FFN width 8192.
- Representative FFN: 1024 → 8192 → ReLU → 1024.
- NLLB tags the direction with `eng_Latn` and `kik_Latn`.
- These are checkpoint-specific values; smaller family members have different configurations.

> Visual: Same layout as the previous slide for direct comparison; 24× badges and NLLB code tags.

## Slide 15 — Language IDs are operational inputs

- Tokenizers and model checkpoints use different language-code conventions.
- mBART example: set `tokenizer.src_lang = "en_XX"`; force a supported target token at generation.
- NLLB example: set `src_lang="eng_Latn"`; force target token `kik_Latn`.
- Wrong, missing, or mismatched IDs can produce the wrong language or degraded output.
- Verify target-code membership in the exact tokenizer vocabulary.

> Visual: Two-column code-token comparison, mBART vs NLLB.

## Slide 16 — Inference pipeline: end to end

- 1. Load the exact checkpoint and its matching tokenizer.
- 2. Set source language and prepare source text.
- 3. Tokenize to IDs and attention mask.
- 4. Generate with the target-language token and explicit decoding settings.
- 5. Decode token IDs; record checkpoint, tokenizer, IDs, and generation parameters.

> Visual: Five numbered compartments with data flowing top to bottom.

## Slide 17 — Minimal Python: mBART-50

- Load `MBart50TokenizerFast` and `MBartForConditionalGeneration` from the same checkpoint.
- Set source language to `en_XX`; use a supported target such as `fr_XX` for this runnable example.
- Pass `forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]` to `generate`.
- Decode with `batch_decode(..., skip_special_tokens=True)`.
- Do not use this checkpoint for English → Kikuyu; the target is unsupported.

> Visual: Syntax-highlighted code excerpt (see generator section in companion notes or repository file).

## Slide 18 — Minimal Python: NLLB-200

- Load `AutoTokenizer` and `AutoModelForSeq2SeqLM` from `facebook/nllb-200-3.3B`.
- Set `src_lang="eng_Latn"` and select `kik_Latn` as target.
- Resolve the target token ID from the matching tokenizer and force it during generation.
- Use inference mode; decode the resulting sequence.
- 3.3B weights require substantial memory; use appropriate hardware and precision.

> Visual: Code-to-pipeline mapping: tokenizer → model.generate → decoded text.

## Slide 19 — Why zero-shot translation fails

- **Coverage gap:** a language may be absent from the model/checkpoint.
- **Data sparsity:** limited or noisy bitext weakens representations for a language or direction.
- **Direction imbalance:** performance can differ between A → B and B → A.
- **Configuration mismatch:** incorrect language ID, tokenizer, or checkpoint.
- **Decoding degeneration:** repetition or unstable output can arise under some model/configuration conditions.

> Visual: Failure taxonomy, grouped into coverage, data, configuration, and decoding.

## Slide 20 — Diagnose before attributing cause

- Confirm the exact model and tokenizer revision.
- Check that both source and target languages are supported by that checkpoint.
- Verify language IDs, script, direction, and generation parameters.
- Inspect raw token output and language identity; test a known supported pair as a control.
- Compare against a vetted reference and consult checkpoint documentation.

> Visual: Decision tree with “supported?” and “IDs correct?” gates before quality analysis.

## Slide 21 — Evaluation: beyond a fluent sentence

- Use suitable automatic metrics (e.g. BLEU or chrF) with documented tokenization and test sets.
- For low-resource languages, consider human evaluation and language-expert review.
- Assess adequacy, fluency, terminology, script, and whether the output is actually in the target language.
- Report test-set provenance, direction, sample size, checkpoint, and decoding configuration.
- A single example illustrates behavior; it is not a benchmark.

> Visual: Evaluation matrix: automatic metrics + human criteria + reproducibility metadata.

## Slide 22 — Reproducibility and responsible reporting

- Pin model and tokenizer identifiers/revisions; note library versions and hardware.
- Record source/target codes, preprocessing, decoding parameters, and random seeds where applicable.
- Disclose unsupported or low-resource directions and uncertainty in references.
- Avoid claiming that parameter count or language count alone caused a result.
- Consider privacy, consent, cultural context, and risks of unreviewed translation.

> Visual: Reproducibility checklist with a responsible-use callout.

## Slide 23 — Translation pipeline vs. RAG pipeline

- Translation is a conditional sequence-generation task: source text → target text.
- The uploaded RAG example has different stages: ingest documents, chunk/embed/index, retrieve relevant context, optionally rerank, then generate an answer.
- Both workflows use models and tokenization, but retrieval is not part of the Transformer translation architecture shown here.
- Keep task pipelines distinct when designing experiments and interpreting outputs.

> Visual: Split diagram: translation path vs. RAG retrieval-and-generation path.

## Slide 24 — Key takeaways

- Zero-shot transfer is possible, not guaranteed.
- Checkpoint coverage is the first diagnostic; mBART-50 excludes Kikuyu, NLLB-200 includes it.
- Language IDs, tokenizer, direction, and generation settings are essential inputs.
- Encoder-decoder anatomy explains information flow; it does not by itself explain every failure.
- Evaluate with suitable data and expertise; report the experiment reproducibly.

> Visual: Five concise takeaway cards; highlight “coverage ≠ quality guarantee.”

## Slide 25 — Discussion and references

- **Discussion:** Which low-resource language pairs matter in your setting, and how will you validate them?
- mBART-50 model card: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
- NLLB-200 3.3B model card: https://huggingface.co/facebook/nllb-200-3.3B
- Vaswani et al. (2017), *Attention Is All You Need*, https://arxiv.org/abs/1706.03762
- Tang et al. (2020), mBART, https://arxiv.org/abs/2008.00401
- NLLB Team et al. (2022), *No Language Left Behind*, https://arxiv.org/abs/2207.04672

> Visual: Questions prompt above a compact reference list.
