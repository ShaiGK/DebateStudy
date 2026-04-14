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

## v2 → v3

### Summary
Two targeted changes based on the v2 pilot rerun: a rewrite of the responsiveness anchors and a tightening of the concession_and_common_ground anchors.

The v2 rewrite of responsiveness over-corrected. v2 required the debater's "position itself" to shift for high scores, but in competitive online debate practically no one changes their overall position. As a result, Claude systematically under-scored responsiveness on the v2 rerun, scoring 2 or 3 where the human annotator scored 4 or 5 in the large majority of cases. The v3 anchors refocus the dimension on tactical reaction to the opponent — new counter-examples, self-corrections, engaging specific objections — while explicitly instructing that maintaining one's overall thesis does not preclude a high score.

Concession also received a minor clarification. v2 fixed the scale (1–3 rather than 1–5) and largely resolved the pilot disagreement, but the v2 anchors did not make the base rate explicit. v3 states plainly that a score of 1 is the default and most common, a score of 2 requires a specific on-the-record concession, and a score of 3 is reserved for debaters whose overall posture is collaborative rather than adversarial and should be assigned at most once across many debates. This is not a scale change; it is a calibration change intended to prevent drift toward 2s-by-default on the next rerun.

### Pilot evidence motivating the change
On the 20-debate v2 rerun:
- Overall agreement moved substantially in the right direction: Cohen's κ on the stated overall judgment rose from 0.245 to 0.474 (3-class) and from 0.331 to 0.733 (2-class, dropping Tie debates). Concession_and_common_ground weighted κ rose from 0.13–0.33 to 0.50–0.58. Respectful engagement and acknowledgment remained stable.
- However, Con responsiveness weighted κ stayed at 0.167 with a bias of −1.50 (Claude rated lower than the human annotator in 19 of 20 cases). The Con-side score distribution was human `{2:1, 3:3, 4:9, 5:7}` vs. Claude `{1:1, 2:8, 3:9, 4:2}` — a full point of median shift, with Claude capping Con responsiveness at 4.
- Qualitative inspection of Claude's v2 justifications for debates 556, 1104, 25047, and 66182 showed Claude describing genuine tactical adaptation — correcting own errors, sharpening contested claims, introducing counter-estimates, engaging specific opponent points — and then assigning scores of 2 or 3 because "the overall framing remains unchanged." Claude was following the v2 rubric exactly as written; the rubric was wrong.

### Changes to the rubric
**responsiveness: anchors refocused on tactical reaction, not positional shift.**
The v2 anchors required "underlying position" or "overall strategy" to change for scores of 4 or 5. The v3 anchors explicitly state that maintaining an overall thesis does not preclude a high score, and reframe the dimension around whether the debater's argumentation visibly reacts to the specific debate (new counter-examples, self-corrections, engaging the opponent's strongest objections). The distinction from acknowledgment is also made explicit: acknowledgment asks whether opponent points are referenced at all; responsiveness asks whether the debater's argumentation is visibly shaped by those points.

**concession_and_common_ground: base-rate guidance added.**
The scale (1–3) and the meta-concession exclusion are unchanged from v2. What is new is explicit base-rate language: 1 is stated to be the default and most common score, 2 requires a specific on-the-record concession, and 3 is described as extremely rare and reserved for debaters with a collaborative rather than adversarial posture. The output rules section is also updated to echo this guidance, so that base-rate information appears both in the rubric definition and in the scoring instructions.

### What did not change
No other dimensions were modified. Acknowledgment, accuracy of representation, and respectful_engagement all produced acceptable agreement on the v2 rerun and are unchanged in v3.

### Validation plan
The v3 rubric will be re-evaluated against the same 20-debate pilot set after Claude is rerun with the updated prompt. Targets: overall stated Cohen's κ ≥ 0.5 (3-class) or ≥ 0.75 (2-class); Con responsiveness weighted κ ≥ 0.35 with bias |Claude − human| < 0.7; concession_and_common_ground agreement maintained at or near v2 levels with no regression on bias. If these targets are met, the rubric will be considered validated for full-scale annotation across the Trimmed dataset.