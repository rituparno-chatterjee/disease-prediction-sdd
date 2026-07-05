# Reflection

Working on this project felt like moving from a rough sketch to a structured pipeline one careful step at a time.

At the beginning, everything looked straightforward: load three datasets, clean them, and prepare them for modeling. In practice, the details mattered a lot more than expected. The same "missing" idea appeared in many different forms (`?`, `na`, `null`, and even noisy tokens like `class` or `meta`), and those inconsistencies could quietly break downstream steps if not handled early.

One of the most useful lessons was that preprocessing order is not just a preference, it is a requirement. Standardizing tokens before label encoding made a big difference because it prevented noisy placeholders from becoming real encoded categories. That single sequencing decision improved consistency across all three datasets.

The kidney dataset was a good reminder that diagnostics are worth the extra effort. Finding and inspecting the dropped target-missing rows made the pipeline feel more trustworthy. Instead of guessing why rows disappeared, we could clearly explain it.

Scaling was another turning point. Moving from single-dataset scaling to a loop across all datasets made the workflow more reusable and less fragile. The extra refinement to exclude ID-like columns was small in code, but important in principle: not every numeric-looking column should be treated as a learnable feature.

There were also practical debugging moments that reinforced good habits. Dtype issues during scaled assignment looked minor at first, but fixing them properly (by rebuilding dataframes safely and preserving column order) made the pipeline more robust.

Overall, the project now feels much more reproducible and intentional:
- the steps are documented,
- outputs are consistent,
- and each transformation has a clear reason.

The logistic baseline phase is now implemented with leakage-safe train/validation/test splitting, dataset-specific preprocessing, artifact persistence, and per-dataset evaluation. The next phase should focus on model calibration, threshold tuning for binary tasks, and stronger comparative baselines.
