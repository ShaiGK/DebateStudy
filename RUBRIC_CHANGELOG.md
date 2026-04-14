# Listening Rubric Changelog

## v1 → v2

### Summary

Version 2 of the listening evaluation rubric introduces three targeted changes motivated by inter-annotator agreement (IAA) data from a 20-debate pilot study comparing a single human annotator against Claude. The most substantive change is a rescaling of the *concession and common ground* dimension from a 1–5 to a 1–3 scale, reflecting the empirical finding that explicit concession is sufficiently rare in online asynchronous debate to render the upper scale points unreliable. Two further changes consist of clarified anchor language: *responsiveness* anchors now explicitly distinguish strategic evolution from point-by-point engagement (the latter belonging to *acknowledgment*), and *respectful engagement* anchors now reserve low scores for genuinely hostile or ad hominem behavior rather than penalizing forceful rhetoric or idea-directed sarcasm. The remaining two dimensions—*acknowledgment* and *accuracy of representation*—were not modified, as their pilot agreement was acceptable and the disagreement patterns did not implicate rubric ambiguity.

### Pilot evidence motivating the change

The v1 pilot comprised 20 debates drawn from the Trimmed dataset, each scored by one human annotator and independently by Claude using the v1 prompt. The most severe reliability problem was in *concession and common ground*: the human annotator's scores were tightly clustered at the low end of the 1–5 scale (distribution: {1: 30, 2: 9, 3: 1} across 40 debater-side observations), whereas Claude used the full range ({1: 5, 2: 17, 3: 13, 4: 4, 5: 1}), producing a per-cell bias of approximately +1.0 to +1.25 (Claude − human) and a weighted κ of only 0.13–0.33 on that dimension. The *responsiveness* dimension showed near-zero weighted κ on the Pro side (0.04); qualitative examination of disagreements indicated that Claude credited any introduction of new content as evidence of adaptation, while the human annotator reserved high scores for debates in which the debater's underlying position or strategic framing demonstrably shifted in response to the opponent's challenges. The *respectful engagement* dimension showed a small but systematic negative bias on the Pro side (−0.35), traceable to Claude penalizing forceful rhetoric and sarcasm directed at ideas rather than at the opponent personally.

### Changes to the rubric

**concession_and_common_ground: rescaled from 1–5 to 1–3.**
The additional scale points were unreliable in the pilot because explicit concession is rare in online asynchronous debate culture, which rewards "winning" over consensus-seeking. The new three-point anchors are: (1) no concession at all—the debater treats every opponent point as wrong or makes only meta-comments outside the debate's substance; (2) at least one explicit acknowledgment of a valid point or shared premise from the opponent; (3) actively seeks common ground, with multiple concessions integrated meaningfully into the argumentation. Meta-statements such as "I am playing devil's advocate" do not count as concession; only concessions of substantive points within the debate count.

**responsiveness: rubric anchors clarified.**
In v1, the anchors for *responsiveness* and *acknowledgment* overlapped conceptually—both rewarded engagement with specific opponent points. The v2 anchors explicitly define *responsiveness* as evolution of the debater's overall argumentative strategy, framing, or position across rounds, and explicitly distinguish this from point-by-point engagement (which is now the sole province of *acknowledgment*). The revised anchors specify that adding new examples or rebuttals while keeping the same underlying framing constitutes only modest adaptation (score 2–3), whereas genuine high-end responsiveness requires the debater's position itself to shift in response to opponent challenges (score 4–5).

**respectful_engagement: calibration guidance added.**
The v1 anchors described score 2 as "frequently dismissive or sarcastic toward the opponent's views," which Claude interpreted as including forceful rhetoric and sarcasm aimed at ideas. The v2 anchors reserve scores of 1 or 2 for genuinely hostile, insulting, ad hominem, or personally dismissive behavior directed at the opponent themselves (not merely at their arguments), and explicitly state that forceful argumentation, idea-directed sarcasm, strong rhetoric, and rhetorical flourishes do not constitute disrespect and should not lower the score. Score 3 is now glossed as "neutral tone; forceful argumentation but not personally disrespectful."

### What did not change

The *acknowledgment* and *accuracy of representation* dimensions were left unchanged. Their pilot agreement was acceptable—weighted κ in the 0.34–0.56 range on the Pro side—and the disagreement patterns did not suggest that the rubric language was responsible for the observed variance. Altering anchors that were functioning adequately risked introducing new sources of inconsistency without a clear evidentiary basis.

### Validation plan

The v2 rubric will be re-evaluated against the same 20-debate pilot set after Claude is rerun with the updated prompt. If pooled within-1 agreement across the four 1–5 dimensions reaches 0.85 or higher and the overall judgment Cohen's κ reaches 0.40 or higher, the rubric will be considered validated for full-scale annotation across the approximately 830 debates in the Trimmed dataset. The concession dimension will be assessed separately on its 1–3 scale; given the scale compression, near-ceiling within-1 agreement is expected and the primary diagnostic will be whether the bias has been eliminated relative to v1.
