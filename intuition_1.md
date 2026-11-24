# Intuition

This Mermaid flowchart places the model-specific observations outside and beneath their corresponding output boxes, styled in red. A white background is explicitly set for the diagram and its nodes.

```mermaid
%%{init: {"theme":"base","themeVariables":{"background":"#ffffff","mainBkg":"#ffffff","primaryColor":"#ffffff","primaryTextColor":"#172033","lineColor":"#596579","fontSize":"16px"},"flowchart":{"htmlLabels":true,"curve":"linear","nodeSpacing":70,"rankSpacing":70}}}%%
flowchart TB
    SOURCE["<div style='width:760px;text-align:center'><b>English Source</b><br/><br/>What is the role of carbohydrates in the diet of dairy cattle?</div>"]

    subgraph COMPARISON["Translation Outputs and Human Reference"]
        direction LR

        subgraph NLLB_COL[" "]
            direction TB
            NLLB["<div style='width:320px;text-align:center'><b>NLLB-200 Output</b><br/><br/>Quel est le rôle des glucides dans l'alimentation du bétail laitier ?</div>"]
            NLLB_NOTE["<div style='width:320px;text-align:center;color:#c62828'><b>Wrong language generated</b></div>"]
            NLLB --> NLLB_NOTE
        end

        subgraph MBART_COL[" "]
            direction TB
            MBART["<div style='width:320px;text-align:center'><b>mBART Output</b><br/><br/>main main main main<br/>main main main main main</div>"]
            MBART_NOTE["<div style='width:320px;text-align:center;color:#c62828'><b>Translation degeneration</b></div>"]
            MBART --> MBART_NOTE
        end

        subgraph HUMAN_COL[" "]
            direction TB
            HUMAN["<div style='width:320px;text-align:center'><b>Human Reference</b><br/><br/>Hihi bata wa irio cia hinya mīrīreinī ya ng'ombe ya iria.</div>"]
        end
    end

    SOURCE --> NLLB
    SOURCE --> MBART
    SOURCE --> HUMAN

    classDef source fill:#ffffff,stroke:#4a78b5,color:#172033,stroke-width:2px;
    classDef model fill:#ffffff,stroke:#d98a42,color:#172033,stroke-width:2px;
    classDef reference fill:#ffffff,stroke:#4d9b61,color:#172033,stroke-width:2px;
    classDef observation fill:#ffffff,stroke:transparent,color:#c62828,stroke-width:0px;

    class SOURCE source;
    class NLLB,MBART model;
    class HUMAN reference;
    class NLLB_NOTE,MBART_NOTE observation;
    style COMPARISON fill:#ffffff,stroke:#8a94a6,color:#172033,stroke-dasharray:5 5
    style NLLB_COL fill:#ffffff,stroke:transparent
    style MBART_COL fill:#ffffff,stroke:transparent
    style HUMAN_COL fill:#ffffff,stroke:transparent
    linkStyle default stroke:#596579,stroke-width:2px
    linkStyle 0,1 stroke:transparent,stroke-width:0px
```
