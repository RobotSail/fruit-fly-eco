#!/usr/bin/env bash
# Auto-generated PostToolUse verification hook
# Compiled from workflow: design

# Read hook payload from stdin (Claude Code passes JSON)
_HOOK_INPUT=$(cat)
_COMMAND=$(echo "$_HOOK_INPUT" | jq -r '.tool_input.command // empty')
PROJECT_PATH="${CLAUDE_PROJECT_DIR:-$PWD}"

[ -z "$_COMMAND" ] && exit 0

# Log every hook invocation
mkdir -p "$PROJECT_PATH/.factory/hooks"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) HOOK_FIRED command=$_COMMAND" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

if echo "$_COMMAND" | grep -q "factory agent researcher"; then
  # Artifact verification: researcher_similar
  _vfail=0
  _f="$PROJECT_PATH/.factory/strategy/research-similar.md"
  [ ! -f "$_f" ] && echo "VERIFY FAIL: researcher_similar: .factory/strategy/research-similar.md missing" && _vfail=1
  [ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: researcher_similar: .factory/strategy/research-similar.md is empty" && _vfail=1
  [ -f "$_f" ] && [ "$(wc -c < "$_f")" -lt 50 ] && echo "VERIFY FAIL: researcher_similar: .factory/strategy/research-similar.md smaller than 50 bytes" && _vfail=1
  [ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=researcher_similar" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
  echo "VERIFY OK: researcher_similar artifacts validated"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=researcher_similar" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

elif echo "$_COMMAND" | grep -q "factory agent researcher"; then
  # Artifact verification: researcher_techstack
  _vfail=0
  _f="$PROJECT_PATH/.factory/strategy/research-techstack.md"
  [ ! -f "$_f" ] && echo "VERIFY FAIL: researcher_techstack: .factory/strategy/research-techstack.md missing" && _vfail=1
  [ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: researcher_techstack: .factory/strategy/research-techstack.md is empty" && _vfail=1
  [ -f "$_f" ] && [ "$(wc -c < "$_f")" -lt 50 ] && echo "VERIFY FAIL: researcher_techstack: .factory/strategy/research-techstack.md smaller than 50 bytes" && _vfail=1
  [ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=researcher_techstack" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
  echo "VERIFY OK: researcher_techstack artifacts validated"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=researcher_techstack" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

elif echo "$_COMMAND" | grep -q "factory agent researcher"; then
  # Artifact verification: researcher_pitfalls
  _vfail=0
  _f="$PROJECT_PATH/.factory/strategy/research-pitfalls.md"
  [ ! -f "$_f" ] && echo "VERIFY FAIL: researcher_pitfalls: .factory/strategy/research-pitfalls.md missing" && _vfail=1
  [ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: researcher_pitfalls: .factory/strategy/research-pitfalls.md is empty" && _vfail=1
  [ -f "$_f" ] && [ "$(wc -c < "$_f")" -lt 50 ] && echo "VERIFY FAIL: researcher_pitfalls: .factory/strategy/research-pitfalls.md smaller than 50 bytes" && _vfail=1
  [ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=researcher_pitfalls" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
  echo "VERIFY OK: researcher_pitfalls artifacts validated"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=researcher_pitfalls" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

elif echo "$_COMMAND" | grep -q "factory agent strategist"; then
  # Artifact verification: strategist
  _vfail=0
  _f="$PROJECT_PATH/.factory/strategy/current.md"
  [ ! -f "$_f" ] && echo "VERIFY FAIL: strategist: .factory/strategy/current.md missing" && _vfail=1
  [ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: strategist: .factory/strategy/current.md is empty" && _vfail=1
  [ -f "$_f" ] && [ "$(wc -c < "$_f")" -lt 200 ] && echo "VERIFY FAIL: strategist: .factory/strategy/current.md smaller than 200 bytes" && _vfail=1
  [ -f "$_f" ] && ! grep -qE '\#\#\#\ Phase\ 1|\#\#\#\ Architecture' "$_f" && echo "VERIFY FAIL: strategist: .factory/strategy/current.md missing required sentinel (### Phase 1, ### Architecture)" && _vfail=1
  [ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=strategist" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
  echo "VERIFY OK: strategist artifacts validated"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=strategist" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

elif echo "$_COMMAND" | grep -q "factory agent builder"; then
  # Artifact verification: builder
  _vfail=0
  _f="$PROJECT_PATH/.factory/reviews/builder-latest.md"
  [ ! -f "$_f" ] && echo "VERIFY FAIL: builder: .factory/reviews/builder-latest.md missing" && _vfail=1
  [ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: builder: .factory/reviews/builder-latest.md is empty" && _vfail=1
  [ -f "$_f" ] && [ "$(wc -c < "$_f")" -lt 500 ] && echo "VERIFY FAIL: builder: .factory/reviews/builder-latest.md smaller than 500 bytes" && _vfail=1
  [ -f "$_f" ] && ! grep -qE 'commit' "$_f" && echo "VERIFY FAIL: builder: .factory/reviews/builder-latest.md missing required sentinel (commit)" && _vfail=1
  [ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=builder" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
  echo "VERIFY OK: builder artifacts validated"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=builder" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

elif echo "$_COMMAND" | grep -q "factory agent health_checker"; then
  # Artifact verification: health_checker
  _vfail=0
  _f="$PROJECT_PATH/.factory/reviews/health-check.md"
  [ ! -f "$_f" ] && echo "VERIFY FAIL: health_checker: .factory/reviews/health-check.md missing" && _vfail=1
  [ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: health_checker: .factory/reviews/health-check.md is empty" && _vfail=1
  [ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=health_checker" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
  echo "VERIFY OK: health_checker artifacts validated"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=health_checker" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

elif echo "$_COMMAND" | grep -q "factory agent code_reviewer"; then
  # Artifact verification: code_reviewer
  _vfail=0
  _f="$PROJECT_PATH/.factory/reviews/code-review.md"
  [ ! -f "$_f" ] && echo "VERIFY FAIL: code_reviewer: .factory/reviews/code-review.md missing" && _vfail=1
  [ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: code_reviewer: .factory/reviews/code-review.md is empty" && _vfail=1
  [ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=code_reviewer" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
  echo "VERIFY OK: code_reviewer artifacts validated"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=code_reviewer" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

elif echo "$_COMMAND" | grep -q "factory agent adversarial_tester"; then
  # Artifact verification: adversarial_tester
  _vfail=0
  _f="$PROJECT_PATH/.factory/reviews/adversarial-qa.md"
  [ ! -f "$_f" ] && echo "VERIFY FAIL: adversarial_tester: .factory/reviews/adversarial-qa.md missing" && _vfail=1
  [ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: adversarial_tester: .factory/reviews/adversarial-qa.md is empty" && _vfail=1
  [ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=adversarial_tester" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
  echo "VERIFY OK: adversarial_tester artifacts validated"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=adversarial_tester" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

elif echo "$_COMMAND" | grep -q "factory agent researcher"; then
  # Artifact verification: graph_explorer
  _vfail=0
  _f="$PROJECT_PATH/.factory/strategy/graph-context.md"
  [ ! -f "$_f" ] && echo "VERIFY FAIL: graph_explorer: .factory/strategy/graph-context.md missing" && _vfail=1
  [ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: graph_explorer: .factory/strategy/graph-context.md is empty" && _vfail=1
  [ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=graph_explorer" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
  echo "VERIFY OK: graph_explorer artifacts validated"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=graph_explorer" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

elif echo "$_COMMAND" | grep -q "factory agent ceo"; then
  # Artifact verification: create_factory_md
  _vfail=0
  _f="$PROJECT_PATH/factory.md"
  [ ! -f "$_f" ] && echo "VERIFY FAIL: create_factory_md: factory.md missing" && _vfail=1
  [ -f "$_f" ] && [ ! -s "$_f" ] && echo "VERIFY FAIL: create_factory_md: factory.md is empty" && _vfail=1
  [ "$_vfail" -ne 0 ] && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_FAIL node=create_factory_md" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt" && exit 1
  echo "VERIFY OK: create_factory_md artifacts validated"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) VERIFY_OK node=create_factory_md" >> "$PROJECT_PATH/.factory/hooks/hook-log.txt"

fi