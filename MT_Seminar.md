# Machine Translation Seminar: End-to-End Workflow

This diagram summarizes the machine-translation workflow from parallel-corpus preparation through model development and evaluation.

```mermaid
%%{init: {"theme": "dark", "flowchart": {"curve": "basis", "htmlLabels": true}, "themeVariables": {"background": "#0b0f14", "primaryColor": "#252b46", "primaryTextColor": "#f3f4f6", "primaryBorderColor": "#a78bfa", "lineColor": "#cbd5e1", "secondaryColor": "#172a1d", "tertiaryColor": "#172436", "clusterBkg": "#151b24", "clusterBorder": "#64748b", "edgeLabelBackground": "#0b0f14", "fontFamily": "Arial, sans-serif"}}}%%
flowchart TB
  subgraph DATA[Dataset Engineering]
    direction LR
    PC[Parallel Corpus] --> DC[Data Collection] --> CL[Data Cleaning] --> TN[Text Normalization] --> TK[Tokenization] --> SPLIT[Train / Validation / Test Split]
  end

  subgraph MODEL[Model Development]
    direction LR
    subgraph FOUNDATION[Foundation Models]
      direction LR
      mBART[mBart]
      MT5[mT5]
      M200[M200]
      NLLB200[NLLB200]
    end

    subgraph FINETUNE[Fine-Tuning Strategies]
      direction LR
      ZERO[Zero-Shot Baseline]
      FEW[Few-Shot Learning]
      SFT[Supervised Fine-Tuning]
      DOMAIN[Domain Adaptation]
    end

    subgraph ARCH[Translation Architecture]
      direction TB
      ENC[Encoder] --> SELF[Self-Attention] --> CROSS[Cross-Attention] --> DEC[Decoder] --> OUT[Translation Output]
    end

    FOUNDATION --> FINETUNE --> ARCH
  end

  DATA --> MODEL

  subgraph EVAL[Evaluation and Analysis]
    direction LR
    subgraph METRICS[Automatic Metrics]
      direction TB
      BLEU[BLEU]
      METEOR[METEOR]
      TER[TER]
      CHRF[chrF]
      COMET[COMET]
    end
    HUMAN[Human Evaluation]
    subgraph ANALYSIS[Analysis]
      direction TB
      ERROR[Error Analysis]
      ABLATION[Ablation Study]
      STATS[Statistical Testing]
    end
    BLEU --> HUMAN
    METEOR --> HUMAN
    TER --> HUMAN
    CHRF --> HUMAN
    COMET --> HUMAN
    HUMAN --> ERROR
    HUMAN --> ABLATION
    HUMAN --> STATS
  end

  MODEL --> EVAL

  classDef node fill:#252b46,stroke:#a78bfa,color:#f3f4f6,stroke-width:1.5px;
  classDef metric fill:#252b46,stroke:#a78bfa,color:#f3f4f6,stroke-width:1.5px;
  classDef human fill:#30264a,stroke:#c4b5fd,color:#ffffff,stroke-width:2px;
  class PC,DC,CL,TN,TK,SPLIT,LLAMA,MISTRAL,GEMMA,PHI,ZERO,INST,SFT,DOMAIN,ENC,SELF,CROSS,DEC,OUT,ERROR,ABlation,STATS node;
  class BLEU,METEOR,TER,CHRF,COMET metric;
  class HUMAN human;
  style DATA fill:#14251a,stroke:#4ade80,color:#f3f4f6,stroke-width:2px
  style MODEL fill:#2b2114,stroke:#fb923c,color:#f3f4f6,stroke-width:2px
  style FOUNDATION fill:#202414,stroke:#a3a322,color:#f3f4f6
  style FINETUNE fill:#202414,stroke:#a3a322,color:#f3f4f6
  style ARCH fill:#202414,stroke:#a3a322,color:#f3f4f6
  style EVAL fill:#142334,stroke:#60a5fa,color:#f3f4f6,stroke-width:2px
  style METRICS fill:transparent,stroke:transparent
  style ANALYSIS fill:transparent,stroke:transparent
```

> Evaluation metrics and human assessment inform error analysis, ablation studies, and statistical testing.
