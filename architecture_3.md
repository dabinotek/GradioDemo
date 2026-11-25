# Compartmentalized Transformer Architectures for Multilingual Translation

This document presents mBART-50 and NLLB-200 as compartmentalized, vertically flowing architectures, with practical inference pipelines and Python examples. Each model separates input preparation, the encoder stack and memory, the decoder stack, and autoregressive output.

The restrained navy, slate, and amber palette, grouped containers, consistent arrows, and distinct operation symbols are designed for an academic-paper figure. Checkpoint dimensions and depths are specific to the named models.

## Visual key

- **Blue hexagon:** multi-head attention.
- **Amber parallelogram:** position-wise fully connected (FC) layer.
- **Gray rounded container:** residual connection and LayerNorm.
- **Cylinder:** final encoder states / memory.
- **Solid arrows:** forward activations; **dashed arrows:** residual or encoder-memory connections.

## mBART-50 many-to-many

mBART-50 has 12 encoder and 12 decoder layers, `d_model=1024`, 16 attention heads, FFN dimension 4096, and GELU activation. English uses the mBART language ID `en_XX`. Kikuyu is not included in mBART-50's 50-language set.

```mermaid
%%{init: {"theme":"base","themeVariables":{"background":"#ffffff","mainBkg":"#ffffff","primaryColor":"#ffffff","primaryTextColor":"#1f2937","lineColor":"#526273","fontSize":"12px"},"flowchart":{"htmlLabels":true,"curve":"linear","nodeSpacing":15,"rankSpacing":18}}}%%
flowchart TB
    subgraph M_INPUT["I · INPUT PREPARATION"]
        direction TB
        M_SENT(["Source sentence<br/>English"]) --> M_LANG["Source language ID<br/><b>en_XX</b>"]
        M_LANG --> M_TOKEN["SentencePiece tokenization"]
        M_TOKEN --> M_EMBED["Token embeddings +<br/>positional embeddings"]
    end

    subgraph M_ENCODER["II · mBART-50 ENCODER · 12 repeated layers · d=1024 · 16 heads · FFN=4096"]
        direction TB
        M_EIN["Encoder layer input"] --> M_EATT{{"Multi-head self-attention<br/>Q, K, V from encoder states"}}
        M_EATT --> M_EADD(["Residual connection<br/>+ LayerNorm"])
        M_EADD --> M_FC1[/"Position-wise FC<br/>1024 → 4096"/]
        M_FC1 --> M_GELU["GELU activation"]
        M_GELU --> M_FC2[/"Position-wise FC<br/>4096 → 1024"/]
        M_FC2 --> M_ENORM(["Residual connection<br/>+ LayerNorm"])
        M_ENORM --> M_EOUT["Encoder layer output"]
        M_EIN -. "residual" .-> M_EADD
        M_EADD -. "residual" .-> M_ENORM
    end
    M_EMBED --> M_EIN

    subgraph M_MEMORY["III · ENCODER MEMORY"]
        direction TB
        M_MEM[("Final encoder states<br/>/ memory")]
        M_KV["Cross-attention keys<br/>and values"]
        M_MEM --> M_KV
    end
    M_EOUT --> M_MEM

    subgraph M_DECODER["IV · mBART-50 DECODER · 12 repeated layers · d=1024 · 16 heads · FFN=4096"]
        direction TB
        M_DINPUT["Shifted-right target tokens<br/>+ target language ID"] --> M_DATT{{"Masked multi-head<br/>self-attention<br/>causal mask"}}
        M_DATT --> M_DADD(["Residual connection<br/>+ LayerNorm"])
        M_DADD --> M_XATT{{"Encoder-decoder<br/>cross-attention"}}
        M_XATT --> M_XADD(["Residual connection<br/>+ LayerNorm"])
        M_XADD --> M_DFC1[/"Position-wise FC<br/>1024 → 4096"/]
        M_DFC1 --> M_DGELU["GELU activation"]
        M_DGELU --> M_DFC2[/"Position-wise FC<br/>4096 → 1024"/]
        M_DFC2 --> M_DNORM(["Residual connection<br/>+ LayerNorm"])
        M_DNORM --> M_DOUT["Decoder layer output"]
        M_DINPUT -. "residual" .-> M_DADD
        M_DADD -. "residual" .-> M_XADD
        M_XADD -. "residual" .-> M_DNORM
    end
    M_KV -. "encoder memory · keys / values" .-> M_XATT

    subgraph M_OUTPUT["V · AUTOREGRESSIVE OUTPUT"]
        direction TB
        M_PROJ[/"Linear vocabulary projection"/] --> M_SOFT["Softmax · next-token probabilities"]
        M_SOFT --> M_NEXT(["Generated target token"])
    end
    M_DOUT --> M_PROJ
    M_NEXT -. "append token and decode again" .-> M_DINPUT

    classDef io fill:#f8fafc,stroke:#64748b,color:#1f2937,stroke-width:1.2px;
    classDef attn fill:#eaf0f6,stroke:#496a8a,color:#1f2937,stroke-width:1.5px;
    classDef ffn fill:#f8f0e5,stroke:#a87945,color:#1f2937,stroke-width:1.3px;
    classDef norm fill:#f1f3f5,stroke:#7b8794,color:#1f2937,stroke-width:1.1px;
    class M_SENT,M_LANG,M_TOKEN,M_EMBED,M_EIN,M_EOUT,M_MEM,M_KV,M_DINPUT,M_DOUT,M_PROJ,M_SOFT,M_NEXT,M_GELU,M_DGELU io;
    class M_EATT,M_DATT,M_XATT attn;
    class M_FC1,M_FC2,M_DFC1,M_DFC2 ffn;
    class M_EADD,M_ENORM,M_DADD,M_XADD,M_DNORM norm;
    style M_INPUT fill:#ffffff,stroke:#73849b,stroke-width:1.5px
    style M_ENCODER fill:#ffffff,stroke:#496a8a,stroke-width:1.8px
    style M_MEMORY fill:#f7f9fb,stroke:#73849b,stroke-width:1.5px
    style M_DECODER fill:#ffffff,stroke:#496a8a,stroke-width:1.8px
    style M_OUTPUT fill:#ffffff,stroke:#73849b,stroke-width:1.5px
```

### mBART-50 inference pipeline

1. Load the translation checkpoint and tokenizer.
2. Set the source language ID (`en_XX`) before tokenization.
3. Tokenize the English sentence to produce input token IDs and attention mask.
4. Generate the target sequence, forcing a **supported** target-language ID.
5. Decode generated token IDs to text.

**Important:** Kikuyu is unsupported by mBART-50. The runnable example uses French (`fr_XX`) as a supported target instead; do not treat this checkpoint as an English-to-Kikuyu model.

```python
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
source_text = "What is the role of carbohydrates in the diet of dairy cattle?"

# Load the checkpoint and its matching tokenizer.
tokenizer = MBart50TokenizerFast.from_pretrained(checkpoint)
model = MBartForConditionalGeneration.from_pretrained(checkpoint)

# mBART-50 language codes are special tokens. Kikuyu is not supported;
# use a supported target such as French for this demonstration.
tokenizer.src_lang = "en_XX"
target_lang = "fr_XX"
inputs = tokenizer(source_text, return_tensors="pt")

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        max_new_tokens=128,
        num_beams=4,
    )

translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print(translation)
```

## NLLB-200 3.3B

NLLB-200 3.3B has 24 encoder and 24 decoder layers, `d_model=1024`, 16 attention heads, FFN dimension 8192, and ReLU activation. The direction is explicitly tagged: English `eng_Latn` → Kikuyu `kik_Latn`. Kikuyu is included in NLLB-200's language set, although coverage alone does not guarantee quality for every direction.

```mermaid
%%{init: {"theme":"base","themeVariables":{"background":"#ffffff","mainBkg":"#ffffff","primaryColor":"#ffffff","primaryTextColor":"#1f2937","lineColor":"#526273","fontSize":"12px"},"flowchart":{"htmlLabels":true,"curve":"linear","nodeSpacing":15,"rankSpacing":18}}}%%
flowchart TB
    subgraph N_INPUT["I · INPUT PREPARATION"]
        direction TB
        N_SENT(["Source sentence<br/>English"]) --> N_LANG["Source language ID<br/><b>eng_Latn</b>"]
        N_LANG --> N_TOKEN["SentencePiece tokenization"]
        N_TOKEN --> N_EMBED["Token embeddings +<br/>positional embeddings"]
    end

    subgraph N_ENCODER["II · NLLB-200 3.3B ENCODER · 24 repeated layers · d=1024 · 16 heads · FFN=8192"]
        direction TB
        N_EIN["Encoder layer input"] --> N_EATT{{"Multi-head self-attention<br/>Q, K, V from encoder states"}}
        N_EATT --> N_EADD(["Residual connection<br/>+ LayerNorm"])
        N_EADD --> N_FC1[/"Position-wise FC<br/>1024 → 8192"/]
        N_FC1 --> N_RELU["ReLU activation"]
        N_RELU --> N_FC2[/"Position-wise FC<br/>8192 → 1024"/]
        N_FC2 --> N_ENORM(["Residual connection<br/>+ LayerNorm"])
        N_ENORM --> N_EOUT["Encoder layer output"]
        N_EIN -. "residual" .-> N_EADD
        N_EADD -. "residual" .-> N_ENORM
    end
    N_EMBED --> N_EIN

    subgraph N_MEMORY["III · ENCODER MEMORY"]
        direction TB
        N_MEM[("Final encoder states<br/>/ memory")]
        N_KV["Cross-attention keys<br/>and values"]
        N_MEM --> N_KV
    end
    N_EOUT --> N_MEM

    subgraph N_DECODER["IV · NLLB-200 3.3B DECODER · 24 repeated layers · d=1024 · 16 heads · FFN=8192"]
        direction TB
        N_DINPUT["Shifted-right target tokens<br/>+ target language ID · <b>kik_Latn</b>"] --> N_DATT{{"Masked multi-head<br/>self-attention<br/>causal mask"}}
        N_DATT --> N_DADD(["Residual connection<br/>+ LayerNorm"])
        N_DADD --> N_XATT{{"Encoder-decoder<br/>cross-attention"}}
        N_XATT --> N_XADD(["Residual connection<br/>+ LayerNorm"])
        N_XADD --> N_DFC1[/"Position-wise FC<br/>1024 → 8192"/]
        N_DFC1 --> N_DRELU["ReLU activation"]
        N_DRELU --> N_DFC2[/"Position-wise FC<br/>8192 → 1024"/]
        N_DFC2 --> N_DNORM(["Residual connection<br/>+ LayerNorm"])
        N_DNORM --> N_DOUT["Decoder layer output"]
        N_DINPUT -. "residual" .-> N_DADD
        N_DADD -. "residual" .-> N_XADD
        N_XADD -. "residual" .-> N_DNORM
    end
    N_KV -. "encoder memory · keys / values" .-> N_XATT

    subgraph N_OUTPUT["V · AUTOREGRESSIVE OUTPUT"]
        direction TB
        N_PROJ[/"Linear vocabulary projection"/] --> N_SOFT["Softmax · next-token probabilities"]
        N_SOFT --> N_NEXT(["Generated Kikuyu token"])
    end
    N_DOUT --> N_PROJ
    N_NEXT -. "append token and decode again" .-> N_DINPUT

    classDef io fill:#f8fafc,stroke:#64748b,color:#1f2937,stroke-width:1.2px;
    classDef attn fill:#eaf0f6,stroke:#496a8a,color:#1f2937,stroke-width:1.5px;
    classDef ffn fill:#f8f0e5,stroke:#a87945,color:#1f2937,stroke-width:1.3px;
    classDef norm fill:#f1f3f5,stroke:#7b8794,color:#1f2937,stroke-width:1.1px;
    class N_SENT,N_LANG,N_TOKEN,N_EMBED,N_EIN,N_EOUT,N_MEM,N_KV,N_DINPUT,N_DOUT,N_PROJ,N_SOFT,N_NEXT,N_RELU,N_DRELU io;
    class N_EATT,N_DATT,N_XATT attn;
    class N_FC1,N_FC2,N_DFC1,N_DFC2 ffn;
    class N_EADD,N_ENORM,N_DADD,N_XADD,N_DNORM norm;
    style N_INPUT fill:#ffffff,stroke:#73849b,stroke-width:1.5px
    style N_ENCODER fill:#ffffff,stroke:#496a8a,stroke-width:1.8px
    style N_MEMORY fill:#f7f9fb,stroke:#73849b,stroke-width:1.5px
    style N_DECODER fill:#ffffff,stroke:#496a8a,stroke-width:1.8px
    style N_OUTPUT fill:#ffffff,stroke:#73849b,stroke-width:1.5px
```

### NLLB-200 3.3B inference pipeline

1. Load the NLLB checkpoint and tokenizer.
2. Set the source language code (`eng_Latn`) before tokenization.
3. Tokenize the source sentence and create model input tensors.
4. Resolve the target language token (`kik_Latn`) and force it as the first generated token.
5. Generate and decode the target sequence.

The 3.3B checkpoint is large and may require a GPU or substantial system memory. The example uses automatic device placement when supported by the installed Transformers/Accelerate versions; use `device_map="cpu"` or move the model and inputs explicitly if needed.

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

checkpoint = "facebook/nllb-200-3.3B"
source_text = "What is the role of carbohydrates in the diet of dairy cattle?"
source_lang = "eng_Latn"
target_lang = "kik_Latn"

# Load tokenizer and model. The 3.3B weights need substantial memory.
tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang=source_lang)
model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

# Ensure this tokenizer's vocabulary contains the requested target language tag.
target_token_id = tokenizer.convert_tokens_to_ids(target_lang)
if target_token_id == tokenizer.unk_token_id:
    raise ValueError(f"Target language code {target_lang!r} is not in this tokenizer")

inputs = tokenizer(source_text, return_tensors="pt")
if not torch.cuda.is_available():
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        forced_bos_token_id=target_token_id,
        max_new_tokens=128,
        num_beams=4,
    )

translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print(translation)
```

## Shared inference pipeline

| Stage | What happens | Model-specific detail |
|---|---|---|
| 1. Load | Load the matching tokenizer and sequence-to-sequence checkpoint. | mBART-50 and NLLB use different language-code conventions. |
| 2. Inject direction | Set the source language ID and select a supported target ID. | mBART-50: `en_XX` / `fr_XX` in the runnable sample. NLLB: `eng_Latn` / `kik_Latn`. |
| 3. Tokenize | Convert source text to token IDs and attention mask. | Use the tokenizer paired with the exact checkpoint. |
| 4. Encode and decode | The encoder creates source states; the decoder cross-attends to them while generating tokens autoregressively. | Implemented by `model.generate(...)`. |
| 5. Constrain output language | Force the target language token at the start of generation. | Use the checkpoint tokenizer's language-token ID. |
| 6. Decode | Convert generated token IDs back to readable text. | `batch_decode(..., skip_special_tokens=True)`. |

## Checkpoint summary

| Checkpoint | Languages | Parameters (approx.) | Encoder / decoder layers | `d_model` / heads / FFN dimension | FFN activation |
|---|---:|---:|---:|---:|---|
| [mBART-50 many-to-many](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) | 50 | 610M | 12 / 12 | 1024 / 16 / 4096 | GELU |
| [NLLB-200 3.3B](https://huggingface.co/facebook/nllb-200-3.3B) | 200 | 3.3B | 24 / 24 | 1024 / 16 / 8192 | ReLU |

NLLB-200 also has smaller checkpoint variants, including distilled 600M and 1.3B models; those have different configurations. The table refers to the named checkpoints.

## Reading the diagrams

The encoder converts source token embeddings into contextual encoder memory. Decoder masked self-attention can access only the target prefix generated so far. Decoder cross-attention uses decoder states as queries and encoder memory as keys and values. The FFN is applied position-wise, using two learned linear projections with an activation between them. The output projection and softmax produce the next-token distribution; generation repeats until the stopping criterion is met.

Language tags specify the requested translation direction but do not add language coverage that the checkpoint did not learn. mBART-50 does not include Kikuyu; NLLB-200 does. The French NLLB output and repeated mBART output are observed examples, not proof of a specific architectural cause. Verify model support, tokenizer, language IDs, generation settings, and reference quality when diagnosing them.

## References

- [mBART-50 many-to-many model card](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)
- [mBART-50 configuration](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/blob/main/config.json)
- [NLLB-200 3.3B model card](https://huggingface.co/facebook/nllb-200-3.3B)
- [NLLB-200 3.3B configuration](https://huggingface.co/facebook/nllb-200-3.3B/blob/main/config.json)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Tang et al., [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://arxiv.org/abs/2008.00401)
- NLLB Team et al., [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672)
