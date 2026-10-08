# Thesis source (corrected)

Full LaTeX source for *"Lightweight ResUMamba-based On-Device Real-Time
Arrhythmia Classification for Wearable ECG Monitoring"* (Ryuha Kim, Dept. of
Computer Science, Yonsei University, February 2027), with the internal
consistency issues found in review applied. `main.pdf` is the compiled
result (pdfLaTeX + BibTeX + kotex, 64 pages, 0 errors, 0 undefined
references after 3 passes).

## Build

```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires `texlive-latex-extra`, `texlive-lang-korean` (kotex), and
`texlive-science` (algorithm/algpseudocode) or equivalent.

## Fixes applied in this pass

Content/consistency fixes (from the internal review):

- Abstract, Ch.1 contributions, and Ch.3.2 now all state the decoder's
  transposed-conv → parameter-free linear-interpolation change (previously
  only in §3.2).
- §3.2 adds one sentence attributing part of the 79.6%-vs-84.9% MAC-reduction
  gap to the (small) interpolation-cost removal, on top of the
  already-stated unreplaced downsampling/shortcut convolutions. The exact
  per-component MAC split could not be independently verified from the
  model code available to this session — see the caveat in the task
  history — so no new numeric decomposition was invented.
- §5.1.1 adds a sentence reconciling Table 5.1's 159 test records with the
  "138 evaluation records" used later: 21 records lacked a valid
  calibration segment and were excluded. This is grounded in the
  159−138=21 arithmetic and the calibration-segment requirement already
  stated in the same section; the exact per-record exclusion criterion
  could not be independently re-run in this session (needs the raw Holter
  `.mat` data), so please double-check this wording against your own
  filtering code before treating it as final.
- Appendix A now states up front that its pruning experiments use
  MIT-BIH/AAMI-EC57-trained models, and are therefore not directly
  comparable to Table 5.3's HiCardi-trained, final-deployment-graph
  latencies.
- Fixed the broken `\ref{sec:bg_scan}` cross-reference in §3.3 (now points
  at §2.3, `sec:bg_mobile_deployment`).
- Fixed `$T_{\mathrm{model}}$`'s cross-reference in the §5.1.3 metric list
  (now points at §5.1.2, `sec:exp_devices`, where it's actually measured).
- §2.1 no longer claims Ch.1 already stated the
  generalization/personalization dichotomy (it doesn't).
- Ch.6's opening sentence no longer implies a validated "real-time
  arrhythmia monitoring" system; reworded to match the abstract's more
  precise framing (beat classification from a streaming ECG, in real time).
- §6.1 (Limitations) adds a paragraph noting that PyTorch↔ExecuTorch
  output-parity was not verified, so Table 5.2's offline performance and
  Tables 5.3/5.4's on-device timing are not linked by a reported parity
  check.
- §5.1.3 adds a caveat that $T_{\mathrm{queue}}$ is reported only as a
  session maximum, not as a time series.
- Fixed a missing-period/dangling-clause grammar bug in §5.2.3 ("...1.0%
  Peak Native Heap...").
- Fixed the stray "F" typo in §2.2 ("...ECG.F\nAs shown...").
- Fig. 3.3's caption now says it depicts a single Res-U block (not the
  full cascaded ResU path).
- Table 3.1 adds an explicit "Cross-attention" row (2.84 M MACs, both
  models) instead of leaving the gap between the module-row sum and the
  printed Total explained only in the caption text.
- §3.1 adds one sentence cross-referencing Appendix A (previously never
  cited from the main text).

Build/toolchain fixes found only by actually compiling (not in the
original review, since no compiler had been run against this source
before):

- `subfigure` package (1995, v2.0) is incompatible with the current
  `caption` package and was unused (no `\subfigure` calls anywhere) —
  removed from `\usepackage{...}` and from `tocbibind`'s `[subfigure]`
  option.
- `grad.sty` defines the `abstract` environment but not
  `abstract_in_korean`, which `abstract_in_korean.tex` needs — added a
  matching definition in `main.tex`.
- `IEEEbib.bst` emits `\url{...}` for some references; `url` package was
  never loaded — added.
- ToC vs. body numbering mismatch (flagged in the review as needing visual
  verification): `tocbibind`'s `numbib` option gave the bibliography a
  chapter number in the body ("Chapter 7") while a separate `titlecontents`
  override hid the number in the ToC ("References" with no number) —
  removed `numbib` so both are unnumbered. Likewise, the appendix's ToC
  `titlecontents` override used `\chaptername` (always "Chapter") instead
  of `\appendixname` ("Appendix"), so the ToC said "Chapter A." while the
  body said "Appendix A." — fixed to use `\appendixname`. Verified in the
  compiled PDF: ToC and body now both say "References" (p. 39) and both
  say "Appendix A." (p. 47).

## Not fixed

Noted in the review but left as-is, either because fixing it needs the
original figure files/visual check this session couldn't do, or because it
is a minor/cosmetic issue with real risk of introducing a new error:
symbol reuse across contexts (`x_t`/`y_t`, `D`/`D_k`), the position of the
"skip coefficient" definition sentence, `y_r` first appearing in a figure
caption rather than the body text, and the exact column assignment in
Fig. 4.3 (depends on `figures/fig_impl_beat.tex`'s TikZ layout, which
wasn't re-verified visually beyond confirming it compiles).
