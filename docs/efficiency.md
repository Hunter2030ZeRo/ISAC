# Efficiency features in the ISAC 3B architecture

The decoder stack combines several mechanisms to reduce the amount of compute that needs
 to be performed on every token:

* **Grouped-query attention (GQA).** Keys and values are projected once per grouped head
  and then expanded to the query head count, which amortises the cost of building the
  KV cache compared to having dedicated key/value projections for every attention head.
* **Token routers for gated attention and DeltaNet.** With the gated paths enabled by
  default, each decoder layer predicts a per-token score. Tokens that fall below the
  configured threshold are masked out so the attention and feed-forward branches can
  skip their matrix multiplies for those positions. Only the routed tokens trigger the
  expensive projections, mirroring the behaviour described in the Qwen3-Next model
  card.
* **Residual gating.** The attention and Gated DeltaNet branches share a lightweight
  squeeze-and-gate module that modulates their residual updates. Because the gates are
  computed with reduced dimensionality projections, they introduce only a small
  overhead while letting the model zero out unneeded updates for masked tokens.
* **Rotary embedding cache.** Cosine and sine tensors are cached per sequence length and
  device so repeated forwards (especially during autoregressive decoding with cache
  reuse) avoid recomputing the RoPE factors.

Together these elements let the model match the routing-oriented efficiency story from
Qwen3-Next while staying compatible with standard Hugging Face generation flows.
