**System Overview**
This system architecture implements a lightweight, multi-stage machine learning pipeline engineered for the de novo generation and screening of antimicrobial peptides to serve as targeted biopesticides.

Generative Pahse:

- **Autoregressive Generation**: Uses a custom Peptide Transformer embedded with a causal self-attention mechanism.

- **Sequence Design:** Designs entirely novel amino acid strings from a vocabulary learned from high-quality peptide databases.

- **Low-Compute Threshold**:Processes sequences linearly on a character level rather than requiring complex three-dimensional structural data inputs, allowing full execution on a local CPU.

Evaluation and Screening Funnel:

- **Parallelized Verification:** Routes raw synthesized candidates through a parallelized evaluation funnel to isolate functional viability and remove non-viable sequences.

- **Motif Extraction:** Features three independent 1D Convolutional Neural Networks optimized with vector-based sliding kernels to capture structural motif embeddings.

- **Tabular Feature Classification:** Pairs the convolutional layers with a 100-estimator Random Forest classifier to evaluate physical parameters including sequence length, amino acid frequencies, and net charge calculations.

- **Validation Threshold**: Validates a candidate sequence only if it satisfies the physical parameters of the random forest gatekeeper and secures an integrated system fitness index score greater than 0.65.
