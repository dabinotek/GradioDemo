# Vertical Transformer Architecture: mBART-50 and NLLB-200

Compact, vertically arranged schematics of the encoder-decoder Transformer used by the named translation checkpoints. The diagrams separate the encoder and decoder stacks, show the canonical sublayer order, and include residual connections, layer normalization, attention, position-wise feed-forward layers, and language-tagged data flow. Each block shown is a representative layer repeated to the depth specified.

> **Architecture scope.** “Exact” here means the standard Transformer layer structure and checkpoint dimensions/depths. Implementation details such as normalization placement, dropout, embedding sharing, and generation configuration can vary by checkpoint and library version. Confirm the linked model configuration when reproducing results.

## Legend

- **Blue hexagon:** multi-head attention (MHA)
- **Amber parallelogram:** position-wise fully connected feed-forward network (FFN)
- **Gray rounded node:** residual addition and layer normalization
- **Dashed arrow:** encoder memory supplied to decoder cross-attention
- **Repeated layer count:** number of encoder or decoder layers in that checkpoint

## mBART-50 many-to-many

This checkpoint has 12 encoder layers and 12 decoder layers, model dimension 1024, 16 attention heads, FFN dimension 4096, and GELU activation. English is identified with `en_XX`. mBART-50 covers 50 languages and does **not** include Kikuyu.

```mermaid
%%{init: {"theme":"base","themeVariables":{"background":"#ffffff","mainBkg":"#ffffff","primaryColor":"#ffffff","primaryTextColor":"#1f2937","lineColor":"#526273","fontSize":"12px"},"flowchart":{"htmlLabels":true,"curve":"linear","nodeSpacing":16,"rankSpacing":20}}}%%
flowchart TB
    M_TEXT(["Source sentence · English"]) --> M_LANG["Prepend source language ID · <b>en_XX</b>"]
    M_LANG --> M_TOK["SentencePiece tokenization"] --> M_EMBED["Token embeddings + positional embeddings"]

    subgraph M_ENCODER["mBART-50 ENCODER · 12 layers · d=1024 · 16 heads · FFN=4096"]
        direction TB
        M_EIN["Encoder layer input"] --> M_EATT{{"Multi-head self-attention<br/>Q, K, V from encoder states"}}
        M_EATT --> M_EADD(["Residual connection + LayerNorm"])
        M_EADD --> M_EFC1[/"Position-wise FC · 1024 → 4096"/]
        M_EFC1 --> M_EACT["GELU"] --> M_EFC2[/"Position-wise FC · 4096 → 1024"/]
        M_EFC2 --> M_ENORM(["Residual connection + LayerNorm"])
        M_ENORM --> M_EOUT["Encoder layer output"]
        M_EIN -. "residual" .-> M_EADD
        M_EADD -. "residual" .-> M_ENORM
    end
    M_EMBED --> M_EIN

    M_EOUT --> M_MEMORY[("Final encoder states / memory")]
    M_MEMORY --> M_DMEM["Cross-attention keys and values"]

    subgraph M_DECODER["mBART-50 DECODER · 12 layers · d=1024 · 16 heads · FFN=4096"]
        direction TB
        M_DIN["Shifted-right target tokens<br/>+ target language ID"] --> M_DATT{{"Masked multi-head self-attention<br/>causal mask · Q, K, V from target"}}
        M_DATT --> M_DADD(["Residual connection + LayerNorm"])
        M_DADD --> M_XATT{{"Encoder-decoder cross-attention<br/>Q from decoder · K/V from encoder"}}
        M_XATT --> M_XADD(["Residual connection + LayerNorm"])
        M_XADD --> M_DFC1[/"Position-wise FC · 1024 → 4096"/]
        M_DFC1 --> M_DACT["GELU"] --> M_DFC2[/"Position-wise FC · 4096 → 1024"/]
        M_DFC2 --> M_DNORM(["Residual connection + LayerNorm"])
        M_DNORM --> M_DOUT["Decoder layer output"]
        M_DIN -. "residual" .-> M_DADD
        M_DADD -. "residual" .-> M_XADD
        M_XADD -. "residual" .-> M_DNORM
    end
    M_DMEM -. "encoder memory" .-> M_XATT
    M_DOUT --> M_PROJ[/"Linear vocabulary projection"/] --> M_SOFT["Softmax / next-token distribution"] --> M_NEXT(["Generated target token"])
    M_NEXT -. "append token and decode again" .-> M_DIN

    classDef io fill:#f8fafc,stroke:#64748b,color:#1f2937,stroke-width:1.2px;
    classDef attn fill:#eaf0f6,stroke:#496a8a,color:#1f2937,stroke-width:1.5px;
    classDef ffn fill:#f8f0e5,stroke:#a87945,color:#1f2937,stroke-width:1.3px;
    classDef norm fill:#f1f3f5,stroke:#7b8794,color:#1f2937,stroke-width:1.1px;
    class M_TEXT,M_LANG,M_TOK,M_EMBED,M_EIN,M_EOUT,M_MEMORY,M_DMEM,M_DIN,M_DOUT,M_PROJ,M_SOFT,M_NEXT,M_EACT,M_DACT io;
    class M_EATT,M_DATT,M_XATT attn;
    class M_EFC1,M_EFC2,M_DFC1,M_DFC2 ffn;
    class M_EADD,M_ENORM,M_DADD,M_XADD,M_DNORM norm;
    style M_ENCODER fill:#ffffff,stroke:#496a8a,stroke-width:1.5px
    style M_DECODER fill:#ffffff,stroke:#496a8a,stroke-width:1.5px
```

## NLLB-200 3.3B

This checkpoint has 24 encoder layers and 24 decoder layers, model dimension 1024, 16 attention heads, FFN dimension 8192, and ReLU activation. The source and target direction is encoded with NLLB language tags: `eng_Latn` → `kik_Latn`. NLLB-200 includes Kikuyu among its supported languages.

```mermaid
%%{init: {"theme":"base","themeVariables":{"background":"#ffffff","mainBkg":"#ffffff","primaryColor":"#ffffff","primaryTextColor":"#1f2937","lineColor":"#526273","fontSize":"12px"},"flowchart":{"htmlLabels":true,"curve":"linear","nodeSpacing":16,"rankSpacing":20}}}%%
flowchart TB
    N_TEXT(["Source sentence · English"]) --> N_LANG["Source language ID · <b>eng_Latn</b>"]
    N_LANG --> N_TOK["SentencePiece tokenization"] --> N_EMBED["Token embeddings + positional embeddings"]

    subgraph N_ENCODER["NLLB-200 3.3B ENCODER · 24 layers · d=1024 · 16 heads · FFN=8192"]
        direction TB
        N_EIN["Encoder layer input"] --> N_EATT{{"Multi-head self-attention<br/>Q, K, V from encoder states"}}
        N_EATT --> N_EADD(["Residual connection + LayerNorm"])
        N_EADD --> N_EFC1[/"Position-wise FC · 1024 → 8192"/]
        N_EFC1 --> N_EACT["ReLU"] --> N_EFC2[/"Position-wise FC · 8192 → 1024"/]
        N_EFC2 --> N_ENORM(["Residual connection + LayerNorm"])
        N_ENORM --> N_EOUT["Encoder layer output"]
        N_EIN -. "residual" .-> N_EADD
        N_EADD -. "residual" .-> N_ENORM
    end
    N_EMBED --> N_EIN

    N_EOUT --> N_MEMORY[("Final encoder states / memory")]
    N_MEMORY --> N_DMEM["Cross-attention keys and values"]

    subgraph N_DECODER["NLLB-200 3.3B DECODER · 24 layers · d=1024 · 16 heads · FFN=8192"]
        direction TB
        N_DIN["Shifted-right target tokens<br/>+ target language ID · <b>kik_Latn</b>"] --> N_DATT{{"Masked multi-head self-attention<br/>causal mask · Q, K, V from target"}}
        N_DATT --> N_DADD(["Residual connection + LayerNorm"])
        N_DADD --> N_XATT{{"Encoder-decoder cross-attention<br/>Q from decoder · K/V from encoder"}}
        N_XATT --> N_XADD(["Residual connection + LayerNorm"])
        N_XADD --> N_DFC1[/"Position-wise FC · 1024 → 8192"/]
        N_DFC1 --> N_DACT["ReLU"] --> N_DFC2[/"Position-wise FC · 8192 → 1024"/]
        N_DFC2 --> N_DNORM(["Residual connection + LayerNorm"])
        N_DNORM --> N_DOUT["Decoder layer output"]
        N_DIN -. "residual" .-> N_DADD
        N_DADD -. "residual" .-> N_XADD
        N_XADD -. "residual" .-> N_DNORM
    end
    N_DMEM -. "encoder memory" .-> N_XATT
    N_DOUT --> N_PROJ[/"Linear vocabulary projection"/] --> N_SOFT["Softmax / next-token distribution"] --> N_NEXT(["Generated Kikuyu token"])
    N_NEXT -. "append token and decode again" .-> N_DIN

    classDef io fill:#f8fafc,stroke:#64748b,color:#1f2937,stroke-width:1.2px;
    classDef attn fill:#eaf0f6,stroke:#496a8a,color:#1f2937,stroke-width:1.5px;
    classDef ffn fill:#f8f0e5,stroke:#a87945,color:#1f2937,stroke-width:1.3px;
    classDef norm fill:#f1f3f5,stroke:#7b8794,color:#1f2937,stroke-width:1.1px;
    class N_TEXT,N_LANG,N_TOK,N_EMBED,N_EIN,N_EOUT,N_MEMORY,N_DMEM,N_DIN,N_DOUT,N_PROJ,N_SOFT,N_NEXT,N_EACT,N_DACT io;
    class N_EATT,N_DATT,N_XATT attn;
    class N_EFC1,N_EFC2,N_DFC1,N_DFC2 ffn;
    class N_EADD,N_ENORM,N_DADD,N_XADD,N_DNORM norm;
    style N_ENCODER fill:#ffffff,stroke:#496a8a,stroke-width:1.5px
    style N_DECODER fill:#ffffff,stroke:#496a8a,stroke-width:1.5px
```

## Checkpoint summary

| Checkpoint | Languages | Parameters (approx.) | Encoder / decoder layers | d_model / heads / FFN dimension | FFN activation |
|---|---:|---:|---:|---:|---|
| [mBART-50 many-to-many](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) | 50 | 610M | 12 / 12 | 1024 / 16 / 4096 | GELU |
| [NLLB-200 3.3B](https://huggingface.co/facebook/nllb-200-3.3B) | 200 | 3.3B | 24 / 24 | 1024 / 16 / 8192 | ReLU |

NLLB-200 also has smaller checkpoint variants, including distilled 600M and 1.3B models. Values above refer only to the named checkpoints; check the exact configuration for reproduction.

## Zero-shot interpretation

Language tags tell the model which direction to translate, while the encoder represents the source and decoder generates target tokens autoregressively. Tags do not create language knowledge absent from training. mBART-50 does not support Kikuyu; NLLB-200 includes it, but coverage alone does not ensure every direction is well represented. The example's French output and repeated mBART output should be interpreted alongside checkpoint support, tokenizer and language IDs, generation settings, and reference quality—not as evidence that one architectural block alone caused the result.

## References

- [mBART-50 many-to-many model card](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)
- [mBART-50 configuration](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/blob/main/config.json)
- [NLLB-200 3.3B model card](https://huggingface.co/facebook/nllb-200-3.3B)
- [NLLB-200 3.3B configuration](https://huggingface.co/facebook/nllb-200-3.3B/blob/main/config.json)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Tang et al., [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://arxiv.org/abs/2008.00401)
- NLLB Team et al., [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672)
