# Runtime Prompt Contract

The runtime uses one `Paint-by-Numbers Orchestrator` with structured `WorkflowPlan` output.

The orchestrator must:

1. Return exactly stages A, B, C, D, and E.
2. Keep the source image as the identity, pose, composition, and coat-color source of truth.
3. State what may change and what must remain invariant in every image-edit prompt.
4. Treat visible facial features and eye catchlights as semantic features, not disposable texture.
5. Treat A and B as flat segmentation assets, not finished paintings: large closed regions, smooth boundaries,
   no brush texture, individual fur, wood grain, terrazzo chips, gradients, or speckles.
6. Require A to use a solid `#FF00FF` background and preserve source-relative canvas geometry.
7. Treat C as deterministic chroma removal, geometric alignment, and pixel compositing; never send C to ImageGen.
8. Introduce restrained oil-paint character only in D, without fragmenting the large-region structure.
9. Optimize E for broad connected regions while keeping a recognizable simplified oil-paint appearance.
10. Never bypass the C or E human approval gates.

The deterministic runtime, rather than the model, owns stage order, attempt limits, artifacts, approvals, 36-color quantization, region merging, numbering, and final exports.
