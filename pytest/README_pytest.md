# SMLP RL Agent Tests

This directory contains tests for the SMLP RL Agent implementation.

## Test Structure

### Unit Tests (`test_rl_agent.py`)
Tests core RL components without LLM dependencies. **Fully deterministic.**

**Components tested:**
- RewardModel: reward computation logic
- ExampleStore: example storage and retrieval
- PromptOptimizer: UCB selection algorithm
- Utility functions: noise filtering, error detection

**Run:**
```bash
pytest test_rl_agent.py -v
```

### Integration Tests (`test_rl_server.py`)
Tests API endpoints with mocked LLM responses. **Deterministic.**

**Endpoints tested:**
- `/generate`: Command generation
- `/feedback`: Feedback submission
- `/execute`: Command execution (dry-run)
- `/stats`: Training statistics
- `/store`: Example store management
- `/config`: Server configuration

**Run:**
```bash
pytest test_rl_server.py -v
```

### End-to-End Tests (`test_e2e_manual.py`)
Full workflow tests with real LLM. **Non-deterministic - manual testing only.**

**Prerequisites:**
1. Start server: `python smlp_agent_rl_server.py --port 8000`
2. Pre-load model: `ollama run deepseek-r1:1.5b "test"`

**Run:**
```bash
python test_e2e_manual.py
```

**Note:** E2E tests may fail due to LLM flakiness. This is expected and acceptable.

## Running All Tests

```bash
# Run deterministic tests (suitable for CI/CD)
pytest test_rl_agent.py test_rl_server.py -v

# Run all tests including manual E2E
pytest test_rl_agent.py test_rl_server.py -v && python test_e2e_manual.py
```

## Test Coverage

| Component | Unit | Integration | E2E |
|-----------|------|-------------|-----|
| RewardModel | ✅ | ✅ | ✅ |
| ExampleStore | ✅ | ✅ | ✅ |
| PromptOptimizer | ✅ | ⚠️ | ✅ |
| Server API | ➖ | ✅ | ✅ |
| LLM Integration | ➖ | ⚠️ (mocked) | ✅ |
| Full Workflow | ➖ | ➖ | ✅ |

✅ = Fully covered  
⚠️ = Partially covered  
➖ = Not applicable

## CI/CD Integration

**Include in CI/CD:**
- `test_rl_agent.py` ✅
- `test_rl_server.py` ✅

**Exclude from CI/CD:**
- `test_e2e_manual.py` ❌ (non-deterministic, requires running server)

## Dependencies

```bash
pip install pytest pytest-cov
```

## Future Extensions

As SMLP grows, add test directories for other modules:
```
pytest/
├── test_rl_agent.py          # RL Agent tests
├── test_rl_server.py          # RL Server tests
├── test_e2e_manual.py         # E2E tests
├── test_smlp_flows.py         # (future) SMLP flows tests
├── test_smlp_agent.py         # (future) Standard agent tests
└── README.md                  # This file
```
