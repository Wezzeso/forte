# Model selection status

Decision on 17 October 2026 (Asia/Qyzylorda): retain the planned untuned baseline as the release fallback. No adapter has earned selection. This is a fallback policy, not proof that weights are available, inference works or the baseline has measured accuracy.

This memo concerns the original human-gold release gate. The user's later authorized automatic first pass separately permits weak training and provisional model selection without manual event annotations; use `evaluation/WEAK-SUPERVISION.md`. Provided binary-label validation does not establish human-reviewed event, fact, timing or legal-grounding accuracy.

The evaluation software has executable synthetic evidence for failure-inclusive whole-video/event diagnostics, timestamp matching, group leakage, grounding gaps and selection gates. Fixtures cannot establish a model's quality. Supplied recordings still require private local human scenario/event/negative/evidence review, approved split lock, and real baseline/candidate predictions under identical conditions. New-angle and actual media prompt-injection behavior remain unmeasured.

After these prerequisites exist, run the local commands in `evaluation/README.md`. Check every completion failure, late result, event FP/FN, unsupported source and evidence-support gap. Select an adapter only if its recorded validation improvement meets the conservative gates. Run final-check once afterward; any changes driven by its errors must disclose reuse as development data. No numeric rubric score or real accuracy is claimed by this memo.
