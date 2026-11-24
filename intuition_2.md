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

## Model Architecture and Zero-Shot Context

The infographic below uses a three-panel layout inspired by the shared reference: (a) the translation challenge, (b) a side-by-side view of the model families and their training differences, and (c) a practical diagnostic path for interpreting zero-shot output. It is conceptual rather than a layer-by-layer specification. Both families use multilingual Transformer encoder-decoder systems; the core architecture is similar, while pretraining, fine-tuning, language coverage, and checkpoint configuration differ.

```mermaid
%%{init: {"theme":"base","themeVariables":{"background":"#ffffff","mainBkg":"#ffffff","primaryColor":"#ffffff","primaryTextColor":"#172033","lineColor":"#596579","fontSize":"14px"},"flowchart":{"htmlLabels":true,"curve":"linear","nodeSpacing":24,"rankSpacing":36}}}%%
flowchart LR
    subgraph A["(a) Challenge: translate into a language without reliable pair-level evidence"]
        direction TB
        A_SRC["English source<br/>Carbohydrates in dairy cattle diet?"] --> A_LANG["Requested target<br/>Confirm the exact language and script"]
        A_LANG --> A_GAP["Coverage gap?<br/>Language, direction, or parallel data may be sparse"]
        A_GAP --> A_OUT["Observed output<br/>Wrong language or repetition"]
    end

    subgraph B["(b) Model families: shared Transformer backbone, different training recipes"]
        direction TB
        subgraph M["mBART / mBART-50"]
            direction LR
            M_IN["Multilingual text<br/>+ language IDs"] --> M_ENC["Shared Transformer<br/>Encoder"] --> M_REP["Contextual<br/>representation"] --> M_DEC["Shared Transformer<br/>Decoder"] --> M_OUT["Target text"]
            M_PRE["Denoising pretraining<br/>reconstruct corrupted text"] -. "pretrains" .-> M_ENC
            M_PRE -. "pretrains" .-> M_DEC
            M_COV["mBART-50: 50 languages<br/>Checkpoint-specific translation fine-tuning"]
        end
        subgraph N["NLLB-200"]
            direction LR
            N_IN["Multilingual text<br/>+ source/target tags"] --> N_ENC["Shared Transformer<br/>Encoder"] --> N_REP["Contextual<br/>representation"] --> N_DEC["Shared Transformer<br/>Decoder"] --> N_OUT["Target text"]
            N_TRAIN["Multilingual bitext<br/>+ translation objective"] -. "trains" .-> N_ENC
            N_TRAIN -. "trains" .-> N_DEC
            N_COV["200-language family<br/>Multiple checkpoint sizes"]
        end
    end

    subgraph C["(c) Interpretation: verify coverage before diagnosing model failure"]
        direction TB
        C_CHECK["1. Check checkpoint support<br/>Is the target language included?"] --> C_CODES["2. Check language codes and script<br/>Are source and target tags correct?"]
        C_CODES --> C_PAIR["3. Check direction and training evidence<br/>Was this pair seen or is transfer required?"]
        C_PAIR --> C_DECODE["4. Inspect decoding and output<br/>Wrong language, repetition, or poor quality?"]
        C_DECODE --> C_RESULT["Conclusion<br/>Language count alone does not guarantee quality"]
    end

    A_OUT -. "compare with both model families" .-> B
    B -. "use these checks" .-> C

    classDef task fill:#ffffff,stroke:#73849b,color:#172033,stroke-width:1.5px;
    classDef shared fill:#eaf1fb,stroke:#4a78b5,color:#172033,stroke-width:1.5px;
    classDef train fill:#fff3e6,stroke:#d98a42,color:#172033,stroke-width:1.5px;
    classDef check fill:#edf6ed,stroke:#4d9b61,color:#172033,stroke-width:1.5px;
    classDef issue fill:#fff0f0,stroke:#c62828,color:#8f1d1d,stroke-width:1.5px;
    class A_SRC,A_LANG,A_GAP task;
    class A_OUT issue;
    class M_IN,M_ENC,M_REP,M_DEC,M_OUT,N_IN,N_ENC,N_REP,N_DEC,N_OUT shared;
    class M_PRE,M_COV,N_TRAIN,N_COV train;
    class C_CHECK,C_CODES,C_PAIR,C_DECODE,C_RESULT check;
    style A fill:#ffffff,stroke:#6f7785,stroke-width:2px,stroke-dasharray:6 4
    style B fill:#ffffff,stroke:#6f7785,stroke-width:2px,stroke-dasharray:6 4
    style C fill:#ffffff,stroke:#6f7785,stroke-width:2px,stroke-dasharray:6 4
    style M fill:#ffffff,stroke:#d98a42,stroke-width:1.5px
    style N fill:#ffffff,stroke:#4a78b5,stroke-width:1.5px
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
