# Intuition

### English Source

> What is the role of carbohydrates in the diet of dairy cattle?

The comparison below is rendered directly in Markdown. The three output/reference boxes are side by side, with the observations underneath the corresponding model output.

| **NLLB-200 Output** | **mBART Output** | **Human Reference** |
|:---:|:---:|:---:|
| Quel est le rôle des glucides dans l'alimentation du bétail laitier ?<br><br>🔴 **Wrong language generated** | main main main main<br>main main main main main<br><br>🔴 **Translation degeneration** | Hihi bata wa irio cia hinya mīrīreinī ya ng'ombe ya iria. |

---

### LaTeX source

The Beamer source is included for use in a LaTeX presentation. GitHub Markdown does not compile Beamer frames or custom environments such as `sourcebox`, `modelbox`, and `referencebox`; the Markdown table above is the rendered preview.

```latex
\begin{frame}{Intuition}

\begin{sourcebox}{English Source}
What is the role of carbohydrates in the diet of dairy cattle?
\end{sourcebox}

\vspace{4mm}

\begin{columns}[T,onlytextwidth]

  \column{0.32\textwidth}
  \begin{modelbox}{NLLB-200 Output}
    \centering
    \vspace*{\fill}
    Quel est le rôle des glucides dans l'alimentation du bétail laitier ?
    \vspace*{\fill}
  \end{modelbox}
  \vspace{2mm}
  \centering
  {\color{red}\small Wrong language generated}

  \column{0.32\textwidth}
  \begin{modelbox}{mBART Output}
    \centering
    \vspace*{\fill}
    main main main main\\
    main main main main main
    \vspace*{\fill}
  \end{modelbox}
  \vspace{2mm}
  \centering
  {\color{red}\small Translation degeneration}

  \column{0.32\textwidth}
  \begin{referencebox}{Human Reference}
    \centering
    \vspace*{\fill}
    Hihi bata wa irio cia hinya mīrīreinī ya ng'ombe ya iria.
    \vspace*{\fill}
  \end{referencebox}

\end{columns}

\end{frame}
```
