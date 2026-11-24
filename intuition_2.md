# Intuition: Why Zero-Shot Translation Can Fail

The examples below compare two model outputs with a human reference. Their output boxes are ordered **mBART, then NLLB-200**; each red observation sits outside and directly beneath its corresponding model output. A white background is specified for the Mermaid diagrams.

```mermaid
%%{init: {"theme":"base","themeVariables":{"background":"#ffffff","mainBkg":"#ffffff","primaryColor":"#ffffff","primaryTextColor":"#172033","lineColor":"#596579","fontSize":"16px"},"flowchart":{"htmlLabels":true,"curve":"linear","nodeSpacing":70,"rankSpacing":70}}}%%
flowchart TB
    SOURCE["<div style='width:760px;text-align:center'><b>English Source</b><br/><br/>What is the role of carbohydrates in the diet of dairy cattle?</div>"]

    subgraph COMPARISON["Translation Outputs and Human Reference"]
        direction LR

        subgraph MBART_COL[" "]
            direction TB
            MBART["<div style='width:320px;text-align:center'><b>mBART Output</b><br/><br/>main main main main<br/>main main main main main</div>"]
            MBART_NOTE["<div style='width:320px;text-align:center;color:#c62828'><b>Translation degeneration</b></div>"]
            MBART --> MBART_NOTE
        end

        subgraph NLLB_COL[" "]
            direction TB
            NLLB["<div style='width:320px;text-align:center'><b>NLLB-200 Output</b><br/><br/>Quel est le rôle des glucides dans l'alimentation du bétail laitier ?</div>"]
            NLLB_NOTE["<div style='width:320px;text-align:center;color:#c62828'><b>Wrong language generated</b></div>"]
            NLLB --> NLLB_NOTE
        end

        subgraph HUMAN_COL[" "]
            direction TB
            HUMAN["<div style='width:320px;text-align:center'><b>Human Reference</b><br/><br/>Hihi bata wa irio cia hinya mīrīreinī ya ng'ombe ya iria.</div>"]
        end
    end

    SOURCE --> MBART
    SOURCE --> NLLB
    SOURCE --> HUMAN

    classDef source fill:#ffffff,stroke:#4a78b5,color:#172033,stroke-width:2px;
    classDef model fill:#ffffff,stroke:#d98a42,color:#172033,stroke-width:2px;
    classDef reference fill:#ffffff,stroke:#4d9b61,color:#172033,stroke-width:2px;
    classDef observation fill:#ffffff,stroke:transparent,color:#c62828,stroke-width:0px;

    class SOURCE source;
    class MBART,NLLB model;
    class HUMAN reference;
    class MBART_NOTE,NLLB_NOTE observation;
    style COMPARISON fill:#ffffff,stroke:#8a94a6,color:#172033,stroke-dasharray:5 5
    style MBART_COL fill:#ffffff,stroke:transparent
    style NLLB_COL fill:#ffffff,stroke:transparent
    style HUMAN_COL fill:#ffffff,stroke:transparent
    linkStyle default stroke:#596579,stroke-width:2px
    linkStyle 0,1 stroke:transparent,stroke-width:0px
```

## Model Architecture (Conceptual)

Both models use multilingual sequence-to-sequence Transformer encoder-decoder architectures. This simplified diagram highlights their different multilingual training setups; it is not a layer-by-layer specification.

```mermaid
%%{init: {"theme":"base","themeVariables":{"background":"#ffffff","mainBkg":"#ffffff","primaryColor":"#ffffff","primaryTextColor":"#172033","lineColor":"#596579","fontSize":"15px"},"flowchart":{"htmlLabels":true,"curve":"linear","nodeSpacing":45,"rankSpacing":55}}}%%
flowchart LR
    subgraph MBART_ARCH["mBART / mBART-50"]
        direction LR
        M_INPUT["Multilingual text<br/>with language IDs"] --> M_ENC["Transformer<br/>Encoder"] --> M_LATENT["Shared contextual<br/>representation"] --> M_DEC["Transformer<br/>Decoder"] --> M_OUTPUT["Target-language<br/>text"]
        M_PRETRAIN["Denoising pretraining:<br/>reconstruct corrupted text"] -. "pretrains" .-> M_ENC
        M_PRETRAIN -. "pretrains" .-> M_DEC
    end

    subgraph NLLB_ARCH["NLLB-200"]
        direction LR
        N_INPUT["Multilingual text<br/>with source/target language tags"] --> N_ENC["Transformer<br/>Encoder"] --> N_LATENT["Shared contextual<br/>representation"] --> N_DEC["Transformer<br/>Decoder"] --> N_OUTPUT["Target-language<br/>text"]
        N_TRAIN["Multilingual bitext training<br/>and translation objective"] -. "trains" .-> N_ENC
        N_TRAIN -. "trains" .-> N_DEC
    end

    classDef block fill:#ffffff,stroke:#607d9b,color:#172033,stroke-width:1.5px;
    classDef training fill:#ffffff,stroke:#8a94a6,color:#172033,stroke-dasharray:4 3;
    class M_INPUT,M_ENC,M_LATENT,M_DEC,M_OUTPUT,N_INPUT,N_ENC,N_LATENT,N_DEC,N_OUTPUT block;
    class M_PRETRAIN,N_TRAIN training;
    style MBART_ARCH fill:#ffffff,stroke:#d98a42,color:#172033,stroke-width:2px
    style NLLB_ARCH fill:#ffffff,stroke:#4a78b5,color:#172033,stroke-width:2px
```

## Model Comparison

| Model family / commonly used checkpoint | Languages covered | Approximate parameters | Training and practical context |
|---|---:|---:|---|
| [mBART-50](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) | 50 | About 610M | Multilingual denoising-pretrained encoder-decoder; the many-to-many checkpoint was fine-tuned for translation across its supported language set. The original mBART-CC25 setup covered 25 languages. |
| [NLLB-200](https://huggingface.co/facebook/nllb-200-3.3B) | 200 | Public checkpoints include 600M (distilled), 1.3B, and 3.3B | Multilingual translation-focused encoder-decoder family trained for broad many-to-many coverage. Use the exact checkpoint's model card to verify its languages and limitations. |

*Parameter counts are checkpoint-specific and approximate. “mBART” and “NLLB-200” each refer to model families, not one single-size model. Language coverage means a language is included in the model's supported set; it does not guarantee equal quality for every language pair or direction.*

## Why Zero-Shot Can Fail

Zero-shot translation asks a model to translate a language pair without direct parallel examples for that exact pair during training or fine-tuning. Shared multilingual representations can enable transfer between languages, but transfer is not guaranteed. If a language is absent from the model's vocabulary or training data—or has sparse, noisy, or mismatched examples—the model may produce the wrong language, repetitive text, or a fluent but inaccurate translation. A large language-count headline alone does not ensure support for a particular language or pair.

For this example, the NLLB-200 output is French rather than the Kikuyu human reference, while mBART degenerates into repeated text. Check the exact checkpoint's supported-language list, the requested source and target language codes, and whether that direction is supported before attributing the result solely to model size or architecture.
