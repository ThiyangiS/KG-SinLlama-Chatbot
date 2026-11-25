# SinLlama generation and verification diagnostic v1

The v4 results reported for this task were 1 accepted, 6 fallbacks, and 2 skipped. The v4 runtime ZIP is available in Git at commit `55c6836`; the evidence packet is still in `results/generation/`. The v4 raw completion log is unavailable here, so the historical diagnoses below use the reported observations. The new GPU run produces a separate audit.

| Question | Reported v4 result | Diagnosis |
| --- | --- | --- |
| Q01 | Accepted an exact supported answer | The reported completion met the source-copy policy. |
| Q03 | Echoed the question | A generation failure. The model did not answer the frescoes and Mirror Wall topics. The generator can stop at an exact question echo and cut it from the cleaned candidate, leaving the verifier with an empty or partial answer. |
| Q05 | Generated unsupported information | A grounding failure in generation. Strict verification rejects a non-source sentence. Experimental verification can recover a complete supported opening only if the unsupported material comes after it and the opening passes its safety checks. The reported result alone does not establish the logged reason or the position of the unsupported material. |

## Controlled generation

- Q01, Q03, and Q05 map to the saved export IDs 'factual', 'multi_topic', and 'mechanism'. Each run logs the packet SHA-256 and retains the original sources and fallback. Retrieval is never invoked.
- Each question runs with bare question text, the existing minimal Sinhala source/question/answer labels, and the existing full Sinhala system prompt. The full profile also uses the longer source label from v4. The minimal-to-full comparison therefore includes that small label change. The bare-question comparison changes both evidence visibility and Q&A formatting.
- All profiles use one 4-bit NF4 SinLlama model, FP16 compute, greedy decoding, repetition penalty 1.0, no n-gram constraint, and the existing EOS, prompt-marker, question-echo, and loop stops. The input cap is 3,000 tokens. Each question shares the same output budget across profiles, computed from the saved evidence even when the prompt omits it. The formula is min(320, max(120, evidence_tokens + 32)).
- The JSONL records the exact prompt, raw decoded continuation, stop reason, generated token count, output budget, cleaned candidate, and generation findings. Both policies verify the same cleaned candidate against the same selected evidence, including in the question-only condition. Each policy logs its reason, verified answer, fallback use, and final answer. If inference fails, verifier fields are null and the row records the error and fallback.

## Offline verifier results

| Manual case | Strict | Experimental | What explains the result |
| --- | --- | --- | --- |
| Q01 exact source answer | Accept | Accept | Complete source match |
| Q01 short paraphrase | Fallback | Fallback | Rewords the sentence beyond the phrase allowlist, with numeric, negation, and uncertainty checks preserved |
| Q03 correct but incomplete | Fallback | Fallback | Covers frescoes only and omits the selected Mirror Wall evidence |
| Q05 unsupported fact | Fallback | Fallback | The pump claim has no selected source match |
| Q05 change in certainty | Fallback | Fallback | Removes the source's uncertainty marker |
| Q05 supported answer then unsupported claim | Fallback | Recover prefix, review | Complete supported claims precede the discarded continuation |
| Q05 unsupported claim then supported answer | Fallback | Fallback | The first claim fails and a safe complete opening is unavailable |
| Q05 allowlisted paraphrase | Fallback | Accept | Existing equivalence preserves the ordered claims and the checked invariants |

From the Sigiriya project root, run the offline matrix and tests, then build the diagnostic package.

```powershell
python -B -m scripts.sinllama_diagnostic_v1 offline
python -B -m unittest discover -s tests
python -B -m scripts.sinllama_diagnostic_v1 bundle
```

The diagnostic notebook is available in Git at commit `55c6836`. On a CUDA runtime with the diagnostic bundle extracted, run `python -B -m scripts.sinllama_diagnostic_v1 gpu`. Check that all nine rows completed inference before comparing profiles.

## Reviewing the outputs

Use [the nine-row CSV](../results/sinllama_diagnostic_v1/manual_review_v1.csv) and assess the raw output before looking at the verifier decision. Record any differences introduced by cleaning. For Q03, mark frescoes and Mirror Wall coverage separately. For Q05, check the water supply, operation by gravity and pressure, and preservation of uncertainty. A supported answer can still fall back under either policy if its wording or coverage fails the policy. A budget stop alone leaves completeness to the reviewer.

| Question | Prompt | Generation issue | Topics covered | All claims supported | Uncertainty preserved | Sinhala readability | Strict reason | Experimental reason | Final origin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q01 / Q03 / Q05 | Bare / Minimal / Full | | | | | | | | |

The offline matrix tests verifier behavior. GPU inference is pending, so the diagnostic does not establish an accuracy change.
