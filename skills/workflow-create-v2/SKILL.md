---
name: workflow-create-v2
description: "Create mode with inference-time scaling — dynamic research, multi-strategy with intent fidelity, workflow-specific QA (mandatory workflow-validate and cli-integration testers), and Overwatch verification. Use when the user says 'create a mode' and wants the v2 pipeline with directors and intent tracking."
disable-model-invocation: true
argument-hint: ""mode description" or "existing_mode: change description""
---

# Create V2 Workflow

The user wants: **$ARGUMENTS**

## Step: Init User Intent

Creates the user intent ledger with the initial mode description.

```bash
python3 -c "from factory.workflow.contributed.create_v2.intent_init import main; main()" "$PROJECT_PATH"
```

### Gate — Has Factory (Automated)

**MANDATORY:** Wait for the preceding agent to finish, then run this check BEFORE spawning the next agent. Do NOT run agents in parallel across this gate.

```bash
python3 -c "from pathlib import Path; exists = Path("$PROJECT_PATH/.factory/config.json").exists(); print("PROCEED" if exists else "HALT")"
```

- **PROCEED** (exit 0 / no FAIL in output) → continue to `graph_update`
- **HALT** (exit non-zero / FAIL in output) → continue to `discover` instead.

## Step: Discover

```bash
factory discover $PROJECT_PATH
```

### Gate — Factory Md Exists (Automated)

**MANDATORY:** Wait for the preceding agent to finish, then run this check BEFORE spawning the next agent. Do NOT run agents in parallel across this gate.

```bash
python3 -c "from pathlib import Path; exists = Path("$PROJECT_PATH/factory.md").exists(); print("PROCEED" if exists else "HALT")"
```

- **PROCEED** (exit 0 / no FAIL in output) → continue to `factory_init`
- **HALT** (exit non-zero / FAIL in output) → continue to `create_factory_md` instead.

## Phase 1: Ceo — Create Factory Md

```bash
factory agent ceo --task "Create factory.md from template. Copy the factory config template to the project root. Fill in: Goal, Scope, Guards, Eval command, Threshold, and Smoke Test. If .factory/eval_spec.json exists, populate the Eval Spec section. If .factory/strategy/current.md has a Research Configuration section, populate research sections (Research Target, Mutable/Fixed Surfaces, etc.).
Read: .factory/eval_profile.json
Write output to: factory.md" --project "$PROJECT_PATH" --timeout 3600
```

```bash
# Artifact verification: create_factory_md
_vfail=0
_f="$PROJECT_PATH/factory.md"
[ ! -f "$_f" ] && echo "VERIFY FAIL: create_factory_md: factory.md missing" && _vfail=1
[ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: create_factory_md: factory.md is empty" && _vfail=1
[ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=create_factory_md" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
echo "VERIFY OK: create_factory_md artifacts validated"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=create_factory_md" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"
```
*(harness verification — DO NOT SKIP)*

## Step: Factory Init

Parse factory.md and generate .factory/config.json. Must run after factory.md is created.

```bash
factory init $PROJECT_PATH
```

## Step: Graph Update

Extract or incrementally update the code knowledge graph before study.

```bash
factory graph update $PROJECT_PATH
```

## Phase 2: Observe

Run local study to gather observations:

```bash
factory study $PROJECT_PATH
```

Writes observations to `.factory/strategy/observations.md`.

If your task includes a focus directive or focus topic, pass it to the study command:
`factory study $PROJECT_PATH --focus "<your focus topic>"`

## Phase 3: Researcher — Graph Explorer

```bash
factory agent researcher --task "Explore the project's code knowledge graph to build structural understanding. Read .factory/strategy/observations.md for focus context.

**Step 0 — detect graph availability:** Your working directory is already the project root. The graph file lives at `$PROJECT_PATH/graph.json` (NOT inside `.factory/`). Run this smoke check FIRST — use a relative path since your CWD is the project root: `test -f graph.json && echo 'GRAPH AVAILABLE' || echo 'NO GRAPH'` — if the output says GRAPH AVAILABLE, proceed with the graph commands below. If the output says NO GRAPH, skip to the fallback section.

**If the graph IS available:**
1. Run `factory graph query "$PROJECT_PATH" "<focus from observations>" --depth 2` to find relevant nodes
2. Run `factory graph explain "$PROJECT_PATH" "<key node>"` on the most important nodes to understand their connections and dependencies
3. Run `factory graph path "$PROJECT_PATH" "<A>" "<B>"` to trace dependency paths between key components
4. Write structured findings to .factory/strategy/graph-context.md covering: key modules and their relationships, dependency paths, architectural layers, entry points and hotspots

**If the graph is NOT available**, fall back to direct file exploration:
1. Use `find . -name '*.py' | head -50` to discover source files
2. Use `grep -rn 'class \|def ' --include='*.py' | head -100` to map functions and classes
3. Use `grep -rn 'import ' --include='*.py' | head -100` to trace dependencies
4. Write the same structured findings to .factory/strategy/graph-context.md
Read: .factory/strategy/observations.md
Write output to: .factory/strategy/graph-context.md" --project "$PROJECT_PATH" --timeout 600
```

```bash
# Artifact verification: graph_explorer
_vfail=0
_f="$PROJECT_PATH/.factory/strategy/graph-context.md"
[ ! -f "$_f" ] && echo "VERIFY FAIL: graph_explorer: .factory/strategy/graph-context.md missing" && _vfail=1
[ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: graph_explorer: .factory/strategy/graph-context.md is empty" && _vfail=1
[ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=graph_explorer" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
echo "VERIFY OK: graph_explorer artifacts validated"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=graph_explorer" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"
```
*(harness verification — DO NOT SKIP)*

## Step: Concat Study

```bash
cat $PROJECT_PATH/.factory/strategy/observations.md $PROJECT_PATH/.factory/strategy/graph-context.md > $PROJECT_PATH/.factory/strategy/study-combined.md
```

## Phase 4: Ceo — Research Director

```bash
factory agent ceo --task "You are the Research Director for this workflow creation session.

Read:
- `.factory/strategy/study-combined.md` — project study findings (observations + graph analysis)
- `.factory/strategy/user-intent.md` — the user's original mode description

Your task has TWO phases:

PHASE 1 — DESIGN RESEARCH DIRECTIONS
Analyze the mode description and identify N orthogonal research dimensions
tailored to workflow construction.

Default dimensions (adapt or replace based on the specific mode):
  - Existing workflow patterns — study `factory/workflow/definitions.py`,
    `factory/workflow/primitives.py`, and contributed workflows to understand
    node types, edge conventions, fork/join patterns, and trigger functions
  - Mode purpose and agent requirements — what agents does this mode need,
    what data flows between them, what gates control quality
  - Workflow design best practices — DAG patterns, quality gate strategies,
    error recovery, reads/writes declarations

You may add dimensions specific to the mode:
  - Integration patterns (for modes that interact with external tools)
  - Security and safety (for modes that run untrusted code)
  - Performance and scaling (for modes with parallel execution)

N is NOT fixed — YOU decide based on mode complexity:
  - Simple mode (single-agent pipeline): 3 directions
  - Medium mode (fork/join, 2-3 agents): 4-5 directions
  - Complex mode (multi-stage, directors, overwatch): 5-7 directions

For each direction, design a TAILORED prompt — not a generic template.
Bad: "Research existing workflow patterns"
Good: "Study the create_workflow() and design-v2 workflow to understand how
       the inherit-and-mutate pattern works: how nodes are added/removed/modified,
       how edges are filtered and extended, and how validate_graph() catches
       wiring errors. Document the exact mutation sequence and common pitfalls."

Write the research plan to `.factory/strategy/research-plan.json`:
```json
[
  {"focus": "...", "slug": "...", "prompt": "..."}
]
```

Constraints:
- Minimum 3 directions, maximum 7
- Each slug must be unique and kebab-case
- Prompts must be specific to THIS mode, not generic templates

PHASE 2 — EXECUTE RESEARCH
For each direction in the plan, spawn a researcher agent:
```
factory agent researcher --task "<direction.prompt>" --project $PROJECT_PATH
```

Each researcher writes to `.factory/strategy/research-<slug>.md`.

After ALL researchers complete, review quality:
- Each research file exists and has substantive content (>50 bytes)
- No two reports cover the same ground excessively
- Key workflow patterns and node types are covered

If a researcher produced thin output, re-invoke it with a more specific prompt.

Write a brief research summary to the end of research-plan.json noting
which directions completed and any quality issues.
Read: .factory/strategy/study-combined.md, .factory/strategy/user-intent.md
Write output to: .factory/strategy/research-plan.json" --project "$PROJECT_PATH" --timeout 3600
```

```bash
# Artifact verification: research_director
_vfail=0
_f="$PROJECT_PATH/.factory/strategy/research-plan.json"
[ ! -f "$_f" ] && echo "VERIFY FAIL: research_director: .factory/strategy/research-plan.json missing" && _vfail=1
[ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: research_director: .factory/strategy/research-plan.json is empty" && _vfail=1
[ -f "$_f" ] && [ "$(wc -c < "$_f")" -lt 20 ] && echo "VERIFY FAIL: research_director: .factory/strategy/research-plan.json smaller than 20 bytes" && _vfail=1
[ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=research_director" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
echo "VERIFY OK: research_director artifacts validated"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=research_director" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"
```
*(harness verification — DO NOT SKIP)*

## Phase 5: Ceo — Strategy Director

```bash
factory agent ceo --task "You are the Strategy Director for this workflow creation session.

Read:
- ALL research reports at `.factory/strategy/research-*.md`
- `.factory/strategy/research-plan.json` — which research directions were explored
- `.factory/strategy/user-intent.md` — the user's original mode description
- `.factory/strategy/study-combined.md` — project context

Your task has TWO phases:

PHASE 1 — DESIGN STRATEGY PERSPECTIVES
Analyze the research findings and user intent to identify M strategy
perspectives for the workflow specification.

Default perspectives (adapt or replace based on the specific mode):
  - Architecture strategy — graph topology, node types, edge wiring,
    fork/join patterns, gate logic, data flow
  - Testing/verification strategy — acceptance criteria, test cases,
    graph validation, SKILL.md generation, CLI integration checks
  - Risk/scope strategy — what to include vs defer, complexity budget,
    mutation ordering, edge case handling

You may add perspectives specific to the mode:
  - Prompt design strategy (for modes with complex agent prompts)
  - Integration strategy (for modes that bridge external systems)
  - Migration strategy (for modes that replace existing workflows)

M is NOT fixed — YOU decide based on mode complexity:
  - Simple mode: 2-3 perspectives
  - Medium mode: 3-4 perspectives
  - Complex mode: 4-5 perspectives

For each perspective, design a TAILORED prompt.
Bad: "Create an architecture strategy for the workflow"
Good: "Design the graph topology for the new mode. The mode needs a
       research director (CEO, 3600s) that dynamically spawns N researchers,
       a strategy director that produces workflow specs with intent fidelity
       checks, and an overwatch for final verification. Use the inherit-and-mutate
       pattern from design-v2: call create_workflow() as the base, add new nodes,
       remove obsolete ones, rewire edges. Specify exact node IDs, types, roles,
       reads/writes, and edge conditions."

Write the strategy plan to `.factory/strategy/strategy-plan.json`:
```json
[
  {"perspective": "...", "slug": "...", "prompt": "..."}
]
```

Constraints:
- Minimum 2 perspectives, maximum 5
- Each slug must be unique and kebab-case
- One perspective MUST cover testing/verification with explicit acceptance criteria
- Prompts must reference specific findings from the research reports

INTENT FIDELITY CHECK (MANDATORY before spawning strategists):
Before writing the strategy plan, extract every distinct ask from
user-intent.md — features, constraints, behaviors, requirements the user
mentioned. Write them as an `"intent_items"` array in strategy-plan.json.
Then verify: does at least one perspective's prompt cover each intent item?
If an intent item is not addressed by any perspective, either add a
perspective or expand an existing prompt to cover it. No user ask may be
silently dropped.

Each strategist prompt MUST include this line at the end:
"IMPORTANT: The user specifically asked for: <list the intent items relevant
to this perspective>. Your strategy MUST address each of these. Do not
substitute your own ideas for what the user asked for."

PHASE 2 — EXECUTE STRATEGIES
For each perspective in the plan, spawn a strategist agent:
```
factory agent strategist --task "<perspective.prompt>" --project $PROJECT_PATH
```

Each strategist writes to `.factory/strategy/strategy-<slug>.md`.

After ALL strategists complete, review quality:
- Each strategy file exists and has substantive content (>100 bytes)
- The testing strategy has a `### Acceptance Criteria` section with checkboxes
- Architecture strategy specifies concrete node IDs, types, and edge wiring
- No critical perspective is missing
- INTENT COVERAGE: re-read user-intent.md and verify every user ask appears
  in at least one strategy output. If a strategist dropped an intent item,
  re-invoke it with explicit instructions to address the missing item.

If a strategist produced thin output, re-invoke it with a more specific prompt.

Write a brief strategy summary to the end of strategy-plan.json noting
which perspectives completed, intent coverage status, and any quality issues.
Read: .factory/strategy/research-plan.json, .factory/strategy/study-combined.md, .factory/strategy/user-intent.md
Write output to: .factory/strategy/strategy-plan.json" --project "$PROJECT_PATH" --timeout 3600
```

```bash
# Artifact verification: strategy_director
_vfail=0
_f="$PROJECT_PATH/.factory/strategy/strategy-plan.json"
[ ! -f "$_f" ] && echo "VERIFY FAIL: strategy_director: .factory/strategy/strategy-plan.json missing" && _vfail=1
[ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: strategy_director: .factory/strategy/strategy-plan.json is empty" && _vfail=1
[ -f "$_f" ] && [ "$(wc -c < "$_f")" -lt 20 ] && echo "VERIFY FAIL: strategy_director: .factory/strategy/strategy-plan.json smaller than 20 bytes" && _vfail=1
[ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=strategy_director" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
echo "VERIFY OK: strategy_director artifacts validated"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=strategy_director" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"
```
*(harness verification — DO NOT SKIP)*

## Phase 6: Strategist — Synthesize Strategy

```bash
factory agent strategist --task "You are the Strategy Synthesizer. Compile one final workflow specification
from all strategy inputs. Your primary obligation is FIDELITY TO USER INTENT —
the spec must capture everything the user asked for.

Read:
- `.factory/strategy/user-intent.md` — ground truth for user's ask (READ THIS FIRST)
- ALL strategy files at `.factory/strategy/strategy-*.md`
- `.factory/strategy/strategy-plan.json` — which perspectives were explored,
  including the `intent_items` array listing every user ask

STEP 1 — INTENT EXTRACTION
Before synthesizing, extract every distinct ask from user-intent.md into a
numbered list. These are the user's requirements. Every single one must
appear in the final spec — either as a feature in the phased plan, an
acceptance criterion, or an explicitly deferred item with rationale.

STEP 2 — SYNTHESIZE
Write the final workflow specification to `.factory/strategy/current.md`.

Required sections (in this order):
### Graph Topology
  The complete DAG: every node ID, type, edges with conditions.
  Use a text-based diagram showing the flow.
### Node Definitions
  For each node: id, type (AgentNode/FnNode/GateNode/ForkNode/JoinNode),
  role (if AgentNode), timeout, reads, writes, post_checks, prompt summary.
### Edge Wiring
  Complete edge list: source → target [condition].
  Highlight RELOOP back-edges and their gate conditions.
### Phased Plan
  #### Phase 1: <name>
  - **What:** <specific implementation steps>
  - **Why:** <rationale>
  - **Acceptance criteria:** <testable criteria>
### Acceptance Criteria
  Full checklist. Each item must be:
  - [ ] Specific enough for pass/fail verification
  - Traceable to user intent (cite which part of user-intent.md)
### MVP Scope
  In vs deferred.
### Deferred Features
  Items explicitly deferred with rationale.

STEP 3 — INTENT COVERAGE AUDIT
After writing current.md, go back to your numbered intent list from Step 1.
For each user ask, verify it appears in the spec:
- In the graph topology as a node or edge, OR
- In acceptance criteria as a testable item, OR
- In deferred features with a rationale for why it's deferred

Write a `### Intent Coverage` section at the end of current.md:
| # | User Ask | Where in Plan | Status |
|---|----------|--------------|--------|
| 1 | <ask> | Node X / Criterion Y / Deferred | Covered / Deferred |

If ANY user ask has status "Missing" — you have failed. Go back and add it
to the appropriate section before finalizing. No user ask may be silently
dropped.

CRITICAL: The ### Acceptance Criteria section is the contract between
the builder and QA. It flows to the QA Director who verifies each
criterion with workflow-specific tests. Make every item testable and unambiguous.
Read: .factory/strategy/strategy-plan.json, .factory/strategy/user-intent.md
Write output to: .factory/strategy/current.md" --project "$PROJECT_PATH" --timeout 600
```

```bash
# Artifact verification: synthesize_strategy
_vfail=0
_f="$PROJECT_PATH/.factory/strategy/current.md"
[ ! -f "$_f" ] && echo "VERIFY FAIL: synthesize_strategy: .factory/strategy/current.md missing" && _vfail=1
[ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: synthesize_strategy: .factory/strategy/current.md is empty" && _vfail=1
[ -f "$_f" ] && [ "$(wc -c < "$_f")" -lt 200 ] && echo "VERIFY FAIL: synthesize_strategy: .factory/strategy/current.md smaller than 200 bytes" && _vfail=1
[ -f "$_f" ] && ! grep -qE '\#\#\#\ Graph\ Topology|\#\#\#\ Node\ Definitions' "$_f" && echo "VERIFY FAIL: synthesize_strategy: .factory/strategy/current.md missing required sentinel (### Graph Topology, ### Node Definitions)" && _vfail=1
[ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=synthesize_strategy" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
echo "VERIFY OK: synthesize_strategy artifacts validated"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=synthesize_strategy" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"
```
*(harness verification — DO NOT SKIP)*

### Steering Point — Strategy (User Approval)

**This is a USER approval gate, NOT a CEO review gate. Do NOT self-approve.**

Present the strategy/findings to the user by summarizing key points in your output.
Then explicitly ask the user: "Do you approve this plan, or do you have feedback?"

**You MUST wait for the user's response before proceeding.**
- The user says "approve", "yes", "looks good", or similar → proceed to next step
- The user provides feedback or corrections → re-run the previous step incorporating their feedback
- Do NOT write a verdict file and auto-proceed — this gate requires human input

*On RELOOP: return to `strategy_director` (max 3 iterations)*

## Phase 7: Archivist Plan

```bash
factory agent archivist --task "Archive the approved workflow specification for the new mode.
Read: .factory/strategy/current.md
Write output to: .factory/archive/create-plan.md" --project "$PROJECT_PATH" --timeout 300 --model haiku &
```
*(fire-and-forget — CEO continues immediately)*

## Phase 8: Builder

```bash
factory agent builder --task "Implement the workflow changes from the approved specification. Read the approved spec at .factory/strategy/current.md. Read CLAUDE.md for project conventions. If the CEO task includes '## Create Mode (Plugin Package)', follow the PLUGIN checklist: 1) Read **output_folder** from the CEO task 2) Create the output directory: mkdir -p <output_folder> 3) Write pyproject.toml with:    name factory-<mode-name>-workflow, version 0.1.0,    build-system hatchling, requires-python >=3.11,    dependencies [remote-factory],    entry point [factory.plugins] <mode-name> = '<mode_name>:register_plugin' 4) Write <mode_name>.py with:    meta dict (name, description),    workflow() function returning a Workflow object,    register_plugin(registry) calling registry.add_modes() and    registry.add_workflow_search_path(str(Path(__file__).parent)) 5) Write README.md with installation and usage 6) Test: pip install -e <output_folder>/ 7) Verify: factory workflow list shows the mode 8) Validate: factory workflow validate <mode-name> 9) Clean up: pip uninstall -y factory-<mode-name>-workflow The plugin package stays in the output directory — do NOT commit it to the factory repo or open a PR. It is a standalone artifact. Do NOT modify factory/workflow/definitions.py or register_all(). Otherwise, if the CEO task includes '## Create Mode (Update Existing Mode)', follow the update checklist: modify the existing workflow function in definitions.py, verify the register_all() entry still resolves, update WORKFLOW_META if needed, verify all 20 registration points from the CEO task, run factory workflow validate <name>, regenerate SKILL.md via factory workflow export-skills, update tests, run pytest and ruff check. Otherwise, follow the new-mode checklist for portable workflows: 1) Create $PROJECT_PATH/.factory/workflows/ directory if it doesn't exist 2) Write the workflow file to $PROJECT_PATH/.factory/workflows/<name>.py 3) The file must contain a `meta` dict with `name` and `description` keys, and a `workflow()` function returning a Workflow object 4) Only import from factory.workflow.primitives and stdlib — no other factory internals 5) Do NOT modify factory/workflow/definitions.py, register_all(), WORKFLOW_META, or CLI wiring — the workflow registry discovers .factory/workflows/ automatically 6) Run factory workflow validate <name> --project-path $PROJECT_PATH to verify the graph 7) Run factory workflow export-skills --project-path $PROJECT_PATH to generate the SKILL.md 8) Write tests in tests/ 9) Run pytest and ruff check to verify Commit changes and open a draft PR.
Read: .factory/strategy/current.md
Write output to: .factory/reviews/builder-latest.md" --project "$PROJECT_PATH" --timeout 1800
```

```bash
# Artifact verification: builder
_vfail=0
_f="$PROJECT_PATH/.factory/reviews/builder-latest.md"
[ ! -f "$_f" ] && echo "VERIFY FAIL: builder: .factory/reviews/builder-latest.md missing" && _vfail=1
[ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: builder: .factory/reviews/builder-latest.md is empty" && _vfail=1
[ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=builder" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
echo "VERIFY OK: builder artifacts validated"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=builder" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"
```
*(harness verification — DO NOT SKIP)*

### CEO Review — Build

Apply the CEO Review Gate protocol:
1. Read the agent output for the preceding step
2. Read artifacts: `.factory/reviews/builder-latest.md`
3. Assess: Read builder output and PR diff. Does work match the approved spec? For plugin packages: verify output directory contains pyproject.toml, workflow .py with meta + workflow() + register_plugin(), and README.md. Verify NO upstream factory files were modified. For new modes: verify workflow file exists at .factory/workflows/<name>.py with meta dict and workflow() function, NOT patched into definitions.py. For existing mode updates: verify definitions.py changes are correct. Tests written. REDIRECT if any component is missing.
4. Write verdict to `.factory/reviews/ceo-verdict-build.md`
5. **PROCEED** → continue to next step
6. **REDIRECT** → re-invoke the preceding agent with corrections (max 2)
7. **ABORT** → log failure and skip to archival

*On RELOOP: return to `builder` (max 3 iterations)*

## Phase 9: Qa (Parallel)

Spawn 3 agents in parallel:

```bash
factory agent health_checker --task "Execute health_checker task for the project.
Read: .factory/reviews/builder-latest.md, .factory/strategy/current.md
Write output to: .factory/reviews/health-check.md" --project "$PROJECT_PATH" --timeout 600 &
```

```bash
factory agent code_reviewer --task "Execute code_reviewer task for the project.
Read: .factory/reviews/builder-latest.md, .factory/strategy/current.md
Write output to: .factory/reviews/code-review.md" --project "$PROJECT_PATH" --timeout 900 &
```

```bash
factory agent ceo --task "You are the QA Director for this workflow creation session.

Read:
- `.factory/strategy/current.md` — the workflow specification with acceptance criteria
- `.factory/strategy/user-intent.md` — what the user ACTUALLY asked for
- `.factory/reviews/builder-latest.md` — what the builder implemented

Your task has THREE phases:

PHASE 1 — DESIGN QA APPROACHES
Analyze the acceptance criteria, the user's intent, and the builder's
implementation to identify K orthogonal testing approaches tailored to
workflow construction.

Default approaches (adapt or replace based on the workflow):
  - Graph structure verification — node count, edge count, node types,
    fork/join target/source matching, gate evaluator types
  - Prompt and data flow verification — reads/writes declarations,
    post_checks, prompt content, artifact paths
  - User intent verification — does the workflow implement what the user
    asked for, not just what the spec says?

You may add approaches specific to the workflow:
  - Registration testing (verify mode appears in CLI, registry, SKILL.md)
  - Inheritance testing (verify base workflow mutations are correct)
  - Edge case testing (missing nodes, dangling edges, circular paths)

K is NOT fixed — YOU decide based on the complexity of the workflow:
  - Simple workflow (3-10 nodes): 2 testers
  - Medium workflow (10-20 nodes): 3 testers
  - Complex workflow (20+ nodes): 4-5 testers

For each approach, design a TAILORED prompt.
Bad: "Test the workflow implementation"
Good: "Verify the create-v2 workflow graph structure: check that fork_qa.targets
       equals join_qa.sources equals ['health_checker', 'code_reviewer', 'qa_director'],
       verify gate_strategy is evaluator_type='user', verify gate_overwatch is
       evaluator_type='agent' with evaluator_role=CEO, check that both archivists
       have blocking=False."

Write the QA plan to `.factory/reviews/qa-plan.json`:
```json
[
  {"approach": "...", "slug": "...", "prompt": "..."}
]
```

Constraints:
- Minimum 2 approaches, maximum 5
- Each slug must be unique and kebab-case
- ALL acceptance criteria from current.md must be covered by at least
  one tester's prompt
- Prompts must reference specific node IDs, edge conditions, and acceptance criteria

MANDATORY WORKFLOW-SPECIFIC TESTERS (hardcoded — always include these):
In addition to your K designed approaches, you MUST always include these
two hardcoded testers. These are non-negotiable.

1. slug: "workflow-validate"
   Prompt: "You are a workflow validation tester. Your ONLY job: run the
   built workflow through the factory's validation pipeline.
   Run these commands and report exact output:
   1. `factory workflow validate <mode-name>` — the graph must validate
      with zero issues
   2. `factory workflow show <mode-name>` — verify node count and edge count
      match the specification
   3. Import the workflow in Python and call validate_graph() directly —
      verify it returns an empty list
   If any command fails or returns unexpected output, that is your finding."

2. slug: "cli-integration"
   Prompt: "You are a CLI integration tester. Your ONLY job: verify the
   new mode integrates correctly with the factory CLI.
   Run these commands and report exact output:
   1. `factory workflow list` — the mode must appear with a non-empty description
   2. `factory workflow show <mode-name>` — must display nodes and edges
   3. `factory workflow export-skills` — must generate SKILL.md for the mode
   4. Verify the SKILL.md file exists at skills/workflow-<mode-name>/SKILL.md
   If any command fails or the mode is missing from output, that is your finding."

**Plugin mode check:** If the CEO task includes '## Create Mode
(Plugin Package)', also include a tester that verifies the plugin package
structure: pyproject.toml with entry-points, workflow .py with meta +
workflow() + register_plugin(), pip install -e succeeds, factory workflow
list shows the mode, factory workflow validate passes, pip uninstall cleanup.
Verify NO upstream factory files were modified.

**Project-local mode check:** For new portable modes, include a tester that
verifies the workflow was written to .factory/workflows/<name>.py (NOT to
definitions.py). Run factory workflow validate <name> --project-path $PROJECT_PATH
and factory workflow show <name> --project-path $PROJECT_PATH. Verify SKILL.md
generated under skills/workflow-<name>/.

These mandatory testers run alongside your K designed testers — they do NOT
count toward K.

PHASE 2 — EXECUTE QA (ALL IN PARALLEL)
Spawn ALL testers in parallel:

For each approach in the plan:
```
factory agent adversarial_tester --review-tag <slug> --task "<approach.prompt>" --project $PROJECT_PATH &
```

Plus the 2 mandatory workflow testers (ALWAYS, non-negotiable):
```
factory agent adversarial_tester --review-tag workflow-validate --task "<prompt>" --project $PROJECT_PATH &
factory agent adversarial_tester --review-tag cli-integration --task "<prompt>" --project $PROJECT_PATH &
```

Then `wait` for all agents to complete.

Each tester writes to `.factory/reviews/adversarial-<slug>-latest.md`.

After ALL agents complete, review quality:
- Each report exists and has substantive findings
- All acceptance criteria are covered by at least one tester
- The workflow-validate and cli-integration testers passed
- No tester missed its assigned focus area

If a tester produced thin output, re-invoke it with a more specific prompt.

Write a brief QA summary to the end of qa-plan.json noting which
approaches completed and any quality issues.
Read: .factory/reviews/builder-latest.md, .factory/strategy/current.md, .factory/strategy/user-intent.md
Write output to: .factory/reviews/qa-plan.json" --project "$PROJECT_PATH" --timeout 3600 &
```

```bash
wait
```

**Important:** Run ALL commands above in a **single** Bash tool call with timeout set to at least 3600 seconds.

```bash
# Artifact verification: health_checker
_vfail=0
_f="$PROJECT_PATH/.factory/reviews/health-check.md"
[ ! -f "$_f" ] && echo "VERIFY FAIL: health_checker: .factory/reviews/health-check.md missing" && _vfail=1
[ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: health_checker: .factory/reviews/health-check.md is empty" && _vfail=1
[ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=health_checker" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
echo "VERIFY OK: health_checker artifacts validated"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=health_checker" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

# Artifact verification: code_reviewer
_vfail=0
_f="$PROJECT_PATH/.factory/reviews/code-review.md"
[ ! -f "$_f" ] && echo "VERIFY FAIL: code_reviewer: .factory/reviews/code-review.md missing" && _vfail=1
[ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: code_reviewer: .factory/reviews/code-review.md is empty" && _vfail=1
[ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=code_reviewer" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
echo "VERIFY OK: code_reviewer artifacts validated"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=code_reviewer" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

# Artifact verification: qa_director
_vfail=0
_f="$PROJECT_PATH/.factory/reviews/qa-plan.json"
[ ! -f "$_f" ] && echo "VERIFY FAIL: qa_director: .factory/reviews/qa-plan.json missing" && _vfail=1
[ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: qa_director: .factory/reviews/qa-plan.json is empty" && _vfail=1
[ -f "$_f" ] && [ "$(wc -c < "$_f")" -lt 20 ] && echo "VERIFY FAIL: qa_director: .factory/reviews/qa-plan.json smaller than 20 bytes" && _vfail=1
[ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=qa_director" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
echo "VERIFY OK: qa_director artifacts validated"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=qa_director" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"
```
*(post-barrier harness verification — DO NOT SKIP)*

## Barrier: Qa

Wait for all parallel agents to complete: `health_checker`, `code_reviewer`, `qa_director`

Read combined outputs: `.factory/reviews/code-review.md`, `.factory/reviews/health-check.md`, `.factory/reviews/qa-plan.json`

## Step: Synthesize Qa

Merges ALL adversarial-*-latest.md reports into one synthesized QA report. Uses glob pattern — works for any K testers. Findings caught by 2+ testers = HIGH confidence. Single-source findings surfaced but marked. Health checker and code reviewer reports included as pass-through.

```bash
python3 -c "from factory.workflow.contributed.create_v2.qa_synthesis import main; main()" "$PROJECT_PATH"
```

### CEO Review — Qa

Apply the CEO Review Gate protocol:
1. Read the agent output for the preceding step
2. Read artifacts: `.factory/reviews/qa-synthesized.md`, `.factory/strategy/user-intent.md`
3. Assess: You are the CEO reviewing QA results for a newly created workflow.

Read:
- `.factory/strategy/user-intent.md` — what the user ACTUALLY asked for
- `.factory/reviews/qa-synthesized.md` — merged QA report (health check,
  code review, and synthesized adversarial findings)

Decision framework:

PROCEED if ALL of these hold:
  1. Workflow validates cleanly (`factory workflow validate` passes)
  2. Mode appears in `factory workflow list` with correct description
  3. SKILL.md was generated successfully
  4. All acceptance criteria from current.md are verified PASS
  5. Health check passes (tests green, lint clean)
  6. No blocking code review issues
  7. No HIGH-confidence adversarial findings that violate user intent

RELOOP to builder (max 3 iterations) if ANY of these hold:
  1. Workflow validation fails — cite the specific issues
  2. Mode missing from registry or CLI
  3. SKILL.md generation failed
  4. An acceptance criterion failed — cite which one
  5. Health check failed — cite which check
  6. Blocking review or HIGH adversarial findings

  When relooping, provide feedback mapped to SPECIFIC user requirements:
  - "User asked for X (user-intent.md), but <finding from QA>"
  - "Acceptance criterion '<criterion>' FAILED: <evidence>"

  IMPORTANT: Append your reloop feedback to .factory/strategy/user-intent.md
  under a new '## [timestamp] Reloop Feedback (Iteration N)' heading.

HALT if:
  - 3 reloops exhausted without resolution
  - Fundamental design flaw that builder iterations cannot fix
4. Write verdict to `.factory/reviews/ceo-verdict-qa.md`
5. **PROCEED** → continue to next step
6. **REDIRECT** → re-invoke the preceding agent with corrections (max 2)
7. **ABORT** → log failure and skip to archival

*On RELOOP: return to `builder` (max 3 iterations)*

## Phase 10: Ceo — Overwatch

```bash
factory agent ceo --task "You are the Overwatch — the final verification agent before the workflow is shown to the user. Your job is to verify that everything the user asked for was actually built and actually tested, with evidence.

You are NOT another QA pass. The QA Director's testers already checked the code. You check the COMPLETENESS and HONESTY of the entire pipeline's output.

Read:
- .factory/strategy/user-intent.md — every ask the user made
- .factory/strategy/current.md — the approved workflow specification
- .factory/reviews/builder-latest.md — what the builder claims to have done
- .factory/reviews/qa-synthesized.md — merged QA report
- .factory/reviews/health-check.md — eval and test results
- .factory/reviews/code-review.md — code review findings

STEP 1 — INTENT CHECKLIST
Extract every distinct user ask from user-intent.md. For each:
- Is it in the workflow graph? (check node IDs, edges, prompts in current.md)
- Is there test evidence in the QA reports? (command + output, not just claims)
- Was the workflow actually validated? (look for `factory workflow validate` output)

STEP 2 — EVIDENCE AUDIT
Read each QA report. For every PASS claim, check:
- Does it show the actual command that was run?
- Does it show the actual output?
- Or is it just "verified — PASS" with no evidence?
Flag every unsupported claim.

STEP 3 — SPOT CHECK (MANDATORY)
Run these concrete validation commands yourself:
1. `factory workflow validate <mode-name>` — the graph must validate cleanly
2. Check SKILL.md existence — `ls skills/workflow-<mode-name>/SKILL.md`
3. `factory workflow list` — the mode must appear in the registry
4. Reads/writes consistency — every file a node reads must be written by
   a predecessor node in the graph. Check the node definitions in current.md.
5. Workflow execution check — import the workflow in Python and verify
   validate_graph() returns an empty list

Show your commands and their output as evidence.

Common agent pitfalls to check for:
- Workflow validates but is missing nodes from the specification
- SKILL.md was generated but the mode doesn't appear in `factory workflow list`
- Tests pass but don't actually test the workflow graph structure
- Node reads a file that no predecessor writes
- Gate has wrong evaluator_type (user vs agent)
- Fork targets don't match join sources
- Edges reference removed nodes

STEP 4 — REPORT
Write a structured report to .factory/reviews/overwatch-latest.md:

# Overwatch Verification Report

## Intent Coverage
| # | User Ask | Built? | Tested? | Evidence? | Status |
|---|----------|--------|---------|-----------|--------|

## Evidence Audit
- Claims with evidence: N
- Claims without evidence: M
- [list unsupported claims]

## Spot Check Results
### Check 1: workflow validate
- Command: <what you ran>
- Output: <what happened>
- Verdict: PASS/FAIL

### Check 2: SKILL.md existence
...

## Verdict
PASS — all user asks verified with evidence
FAIL — [list what's missing or unsupported]
Read: .factory/reviews/builder-latest.md, .factory/reviews/code-review.md, .factory/reviews/health-check.md, .factory/reviews/qa-synthesized.md, .factory/strategy/current.md, .factory/strategy/user-intent.md
Write output to: .factory/reviews/overwatch-latest.md" --project "$PROJECT_PATH" --timeout 1800
```

```bash
# Artifact verification: overwatch
_vfail=0
_f="$PROJECT_PATH/.factory/reviews/overwatch-latest.md"
[ ! -f "$_f" ] && echo "VERIFY FAIL: overwatch: .factory/reviews/overwatch-latest.md missing" && _vfail=1
[ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: overwatch: .factory/reviews/overwatch-latest.md is empty" && _vfail=1
[ -f "$_f" ] && [ "$(wc -c < "$_f")" -lt 100 ] && echo "VERIFY FAIL: overwatch: .factory/reviews/overwatch-latest.md smaller than 100 bytes" && _vfail=1
[ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=overwatch" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
echo "VERIFY OK: overwatch artifacts validated"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=overwatch" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"
```
*(harness verification — DO NOT SKIP)*

### CEO Review — Overwatch

Apply the CEO Review Gate protocol:
1. Read the agent output for the preceding step
2. Read artifacts: `.factory/reviews/overwatch-latest.md`, `.factory/strategy/user-intent.md`
3. Assess: You are the CEO reviewing the Overwatch verification report for a new workflow.

Read:
- .factory/reviews/overwatch-latest.md — the Overwatch's findings
- .factory/strategy/user-intent.md — what the user asked for

The Overwatch has verified whether everything the user asked for was actually built and tested with evidence.

PROCEED if:
- All user asks in the Intent Coverage table show Status = Covered
- No unsupported claims in the Evidence Audit
- All spot checks passed (especially workflow validate and SKILL.md existence)
- The Overwatch verdict is PASS

RELOOP to builder if:
- Any user ask is missing or untested
- There are unsupported QA claims (tests claimed to pass without evidence)
- Spot checks failed (workflow doesn't validate, SKILL.md missing, mode not in registry)
- The Overwatch verdict is FAIL

When relooping, include the specific Overwatch findings in your feedback:
- Which user asks are missing
- Which claims lack evidence
- Which spot checks failed and what the output was

The builder will fix the issues and the full QA + Overwatch pipeline will re-run.
4. Write verdict to `.factory/reviews/ceo-verdict-overwatch.md`
5. **PROCEED** → continue to next step
6. **REDIRECT** → re-invoke the preceding agent with corrections (max 2)
7. **ABORT** → log failure and skip to archival

*On RELOOP: return to `builder` (max 3 iterations)*

### CEO Review — Doc Freshness

Apply the CEO Review Gate protocol:
1. Read the agent output for the preceding step
2. Read artifacts: `.factory/reviews/qa-synthesized.md`
3. Assess: Check the PR diff for documentation freshness. If public APIs, CLI commands, configuration options, or architecture were changed or added, corresponding documentation (README.md, CLAUDE.md, docstrings, --help text, or doc/ files) MUST be updated. PROCEED if docs are current or no doc-worthy changes exist. RELOOP to builder if documentation is stale — specify exactly which changes need doc updates.
4. Write verdict to `.factory/reviews/ceo-verdict-doc-freshness.md`
5. **PROCEED** → continue to next step
6. **REDIRECT** → re-invoke the preceding agent with corrections (max 2)
7. **ABORT** → log failure and skip to archival

*On RELOOP: return to `builder` (max 3 iterations)*

### Gate — Precheck (Automated)

**MANDATORY:** Wait for the preceding agent to finish, then run this check BEFORE spawning the next agent. Do NOT run agents in parallel across this gate.

```bash
factory precheck $PROJECT_PATH --score-before 0 --score-after 0
```

- **PROCEED** (exit 0 / no FAIL in output) → continue to `archivist_build`
- **HALT** (exit non-zero / FAIL in output) → continue to `archivist_build` instead.

## Phase 11: Archivist Build

```bash
factory agent archivist --task "Archive the new mode build results and learnings.
Read: .factory/reviews/qa-synthesized.md
Write output to: .factory/archive/create-build.md" --project "$PROJECT_PATH" --timeout 300 --model haiku &
```
*(fire-and-forget — CEO continues immediately)*
