"""Flare Analyst: Claude (via Amazon Bedrock, inside AWS) answering questions with the portal's own analyses.

Claude never sees cells or raw data, only the small summary results the tools return.
"""
import math
import os

import boto3

MODELS = {
    "Claude Sonnet 4.6 (fast)": "us.anthropic.claude-sonnet-4-6",
    "Claude Opus 4.6 (deepest)": "us.anthropic.claude-opus-4-6-v1",
}
REGION = os.environ.get("AWS_REGION", "us-east-1")

SYSTEM = """You are the Flare Analyst for a hackathon project on ImYoo's study of inflammatory bowel disease (IBD).

The data: 29 people collected blood at home, both during flares and when feeling fine (133 samples, ~427,000
immune cells, single-cell RNA sequencing). 23 have bowel disease (Crohn's, ulcerative colitis); 5 have other
conditions (arthritis, lupus-related, fibromyalgia). People appear only as P01-P29.

How the analyses work:
- Main comparisons put each person's flare samples against their OWN feeling-fine samples, then ask whether
  the change points the same way for most people (Wilcoxon signed-rank test, Benjamini-Hochberg "adjusted p").
- Something "passes the strict test" only if adjusted p < 0.05 (genes: adjusted p < 0.1).
- The team's earlier look suggested B cells (especially IgM memory B cells) rise during flares, and that
  comparing people with their own normal works better than one rule for everyone. Treat these as claims to
  check, not facts.
- Steroids treat flares and change immune cells; use exclude_within_hours_of_steroids to check whether a result
  is driven by steroids rather than the flare itself. Steroid timing was AI-extracted from patients' answers
  without human checking: a recorded time is probably right, but a blank may hide unreported or missed steroid
  use. So say "recorded steroid use", and never claim steroids are fully ruled out.

Rules:
- ONLY state numbers that come from tool results. Never invent or estimate numbers.
- Tool results always win. If they contradict anything above, say so plainly and trust the tool results.
- Don't speculate about causes beyond what the tools show. If a result looks puzzling, say it's puzzling and
  suggest a check that could explain it.
- Always say how many people, how many went up, the adjusted p, and whether it passes the strict test.
  Write very small p-values as "below 0.001", never "0.000".
- Be honest and skeptical: say "trend" or "no clear change" when it doesn't pass; mention small numbers.
- Speak plainly to a non-expert: no jargon, or explain it in a few words.
- Keep each answer short (under 150 words). Use the tools rather than guessing; call list_options first if
  you are unsure of exact cell type, gene or batch names.
- Never try to identify anyone.
- End every answer with one line: "Next question: ..." suggesting the most useful follow-up.
"""

AUTOPILOT = """Explore the data on your own for {rounds} rounds, like a scientist. For each round write:
**Hypothesis:** one sentence. Then test it with the tools. Then **Result:** the numbers in plain words.
Then **Next question:** what this suggests, and use it as the next round's hypothesis.
Start from what is already known (B cells rise during flares) and look for something new or a weakness.
Finish with a 3-sentence **Summary** of what held up and what didn't."""

TOOLS = [
    {"name": "list_options",
     "description": "Exact names of cell types (per detail level), cell types with gene results, lab batches, "
                    "and disease labels. Call this when unsure of a name.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "study_overview",
     "description": "Counts of people, samples, flare vs feeling-fine samples, people with both, and conditions.",
     "input_schema": {"type": "object", "properties": {
         "group": {"type": "string", "enum": ["Bowel disease", "Everyone", "Other"]}}, "required": ["group"]}},
    {"name": "compare_cell_types",
     "description": "Which cell types change during a flare (each person vs their own calm samples). Returns every "
                    "cell type with average change, people up, adjusted p. Options: leave out one lab batch, keep "
                    "only people whose samples were all in one batch, or restrict to a disease (e.g. 'Crohn', "
                    "'Ulcerative').",
     "input_schema": {"type": "object", "properties": {
         "level": {"type": "string", "enum": ["family", "broad", "detailed", "tcell"],
                   "description": "family = whole families like B Cells (subtypes added together); broad = main cell "
                                  "types (in the newest data e.g. Naive B Cells); detailed = finest types; tcell = "
                                  "T-cell subsets (older data only). Use list_options to see which exist."},
         "group": {"type": "string", "enum": ["Bowel disease", "Everyone", "Other"]},
         "leave_out_batch": {"type": "string", "description": "batch name like R99, or omit"},
         "same_batch_only": {"type": "boolean"},
         "disease": {"type": "string", "description": "text the disease label must contain, e.g. Crohn"},
         "exclude_within_hours_of_steroids": {"type": "integer", "description": "leave out samples taken within "
                                              "this many hours of RECORDED steroids, e.g. 168 for a week, or a huge "
                                              "number (100000) for any recorded use. Records may be incomplete."}},
         "required": ["level", "group"]}},
    {"name": "cell_type_person_by_person",
     "description": "For one cell type: each person's share when feeling fine vs during flare (draws one line per person).",
     "input_schema": {"type": "object", "properties": {
         "level": {"type": "string", "enum": ["family", "broad", "detailed", "tcell"]},
         "cell_type": {"type": "string"},
         "group": {"type": "string", "enum": ["Bowel disease", "Everyone", "Other"]}},
         "required": ["level", "cell_type", "group"]}},
    {"name": "gene_changes",
     "description": "Genes that switch on or off during flares inside one broad cell type (bowel-disease people, "
                    "each vs themselves). Returns the top genes with change, people up, adjusted p.",
     "input_schema": {"type": "object", "properties": {
         "cell_type": {"type": "string", "description": "broad cell type, e.g. B Cells"},
         "direction": {"type": "string", "enum": ["up", "down", "both"]},
         "top_n": {"type": "integer", "minimum": 3, "maximum": 25}},
         "required": ["cell_type"]}},
    {"name": "flare_score",
     "description": "How well a blood score tells flare from calm samples ('correct ranking': 50% = coin toss), "
                    "and whether calm samples shortly before a flare already look flare-like.",
     "input_schema": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["general", "personal_bcell", "personal_all"]},
         "group": {"type": "string", "enum": ["Bowel disease", "Everyone", "Other"]},
         "window_days": {"type": "integer", "minimum": 7, "maximum": 120}},
         "required": ["kind", "group"]}},
    {"name": "red_flag",
     "description": "How many flare and calm samples have B cells more than `limit` percentage points above "
                    "that person's own normal, and how many different people the flagged flares come from.",
     "input_schema": {"type": "object", "properties": {
         "limit": {"type": "number", "minimum": 1, "maximum": 50},
         "group": {"type": "string", "enum": ["Bowel disease", "Everyone", "Other"]}},
         "required": ["limit", "group"]}},
]


def clean(x):
    """Make tool results JSON-safe and short (round floats, drop NaN)."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float):
        return None if math.isnan(x) else float(f"{x:.4g}")  # keeps tiny p-values like 3.2e-05
    return x


REPORT_SYSTEM = """You write a short, warm, plain-language note to a patient about their own research blood report.
Use ONLY the facts given. No jargon (explain any term in a few words). Under 120 words.
Never diagnose, predict, or give medical advice; suggest talking to their care team about how they feel.
Say clearly this is a research prototype from a small study (29 people)."""


def explain_report(bedrock, model_id, facts):
    """One plain-language paragraph for the patient report, from the report's numbers only."""
    resp = bedrock.converse(
        modelId=model_id, system=[{"text": REPORT_SYSTEM}],
        messages=[{"role": "user", "content": [{"text": "Report facts:\n" + "\n".join(f"- {f}" for f in facts)}]}],
        inferenceConfig={"maxTokens": 400, "temperature": 0.3})
    return "".join(b.get("text", "") for b in resp["output"]["message"]["content"])


def client():
    return boto3.client("bedrock-runtime", region_name=REGION)


def ask(bedrock, model_id, messages, tools_impl, on_event, max_steps=12):
    """Run the tool-use loop until Claude answers without calling a tool.

    messages: Bedrock Converse message list (mutated in place).
    tools_impl: name -> function(**input) returning (result_dict, figure_or_None).
    on_event: callback(kind, payload) with kind in {"text", "tool"}.
    """
    spec = {"tools": [{"toolSpec": {"name": t["name"], "description": t["description"],
                                    "inputSchema": {"json": t["input_schema"]}}} for t in TOOLS]}
    for _ in range(max_steps):
        resp = bedrock.converse(modelId=model_id, system=[{"text": SYSTEM}], messages=messages,
                                toolConfig=spec, inferenceConfig={"maxTokens": 2000, "temperature": 0.2})
        msg = resp["output"]["message"]
        messages.append(msg)
        results = []
        for block in msg["content"]:
            if block.get("text"):
                on_event("text", block["text"])
            if "toolUse" in block:
                use = block["toolUse"]
                try:
                    result, fig = tools_impl[use["name"]](**(use.get("input") or {}))
                    status = "success"
                except Exception as e:  # tell Claude what went wrong so it can retry with better input
                    result, fig, status = {"error": f"{type(e).__name__}: {e}"}, None, "error"
                result = clean(result)
                on_event("tool", {"name": use["name"], "input": use.get("input") or {}, "result": result, "fig": fig})
                results.append({"toolResult": {"toolUseId": use["toolUseId"], "status": status,
                                               "content": [{"json": result}]}})
        if resp["stopReason"] != "tool_use":
            return
        messages.append({"role": "user", "content": results})
    on_event("text", "_(Stopped after the maximum number of analysis steps.)_")


def describe_call(name, args):
    """Short plain-language label for a tool call, shown in the chat."""
    labels = {
        "list_options": "Looking up exact names",
        "study_overview": "Counting people and samples",
        "compare_cell_types": "Comparing cell types, flare vs feeling fine",
        "cell_type_person_by_person": "Checking person by person",
        "gene_changes": "Looking at genes",
        "flare_score": "Testing a flare score",
        "red_flag": "Checking the red flag",
    }
    details = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{labels.get(name, name)}" + (f" ({details})" if details else "")

