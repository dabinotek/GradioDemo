# Intuition — LaTeX

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
