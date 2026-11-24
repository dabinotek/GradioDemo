# Intuition

This Mermaid flowchart shows the English source above the three side-by-side translation outputs and human reference. The model observations appear beneath their respective outputs.

```mermaid
flowchart TB
    SOURCE["English Source<br/><br/>What is the role of carbohydrates in the diet of dairy cattle?"]

    subgraph COMPARISON["Translation Outputs and Human Reference"]
        direction LR
        NLLB["NLLB-200 Output<br/><br/>Quel est le rôle des glucides dans l'alimentation du bétail laitier ?<br/><br/>🔴 Wrong language generated"]
        MBART["mBART Output<br/><br/>main main main main<br/>main main main main main<br/><br/>🔴 Translation degeneration"]
        HUMAN["Human Reference<br/><br/>Hihi bata wa irio cia hinya mīrīreinī ya ng'ombe ya iria."]
    end

    SOURCE --> NLLB
    SOURCE --> MBART
    SOURCE --> HUMAN

    classDef source fill:#e8f1ff,stroke:#4a78b5,color:#172033,stroke-width:2px;
    classDef model fill:#fff3e6,stroke:#d98a42,color:#172033,stroke-width:2px;
    classDef reference fill:#eaf7ed,stroke:#4d9b61,color:#172033,stroke-width:2px;

    class SOURCE source;
    class NLLB,MBART model;
    class HUMAN reference;
    style COMPARISON fill:transparent,stroke:#8a94a6,color:#172033,stroke-dasharray:5 5
    linkStyle default stroke:#596579,stroke-width:2px
```
