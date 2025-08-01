# Terragon Autonomous SDLC System

This directory contains the Terragon autonomous SDLC enhancement system that continuously discovers and executes the highest-value work for optimal repository maturity.

## System Components

### 🔍 Value Discovery Engine (`value-discovery.py`)
- **Multi-source signal harvesting** from git history, static analysis, security scans
- **WSJF + ICE + Technical Debt scoring** for comprehensive value assessment  
- **Adaptive prioritization** based on repository maturity level
- **Automated backlog generation** with detailed metrics

### ⚡ Autonomous Executor (`autonomous-executor.py`)
- **Intelligent work selection** based on composite scoring
- **Category-specific execution** (security, tech debt, code quality, etc.)
- **Comprehensive validation** with automated rollback
- **Pull request automation** with detailed change tracking

### 📅 Scheduler (`schedule.sh`)
- **Continuous execution cycles** based on value triggers
- **Multiple execution modes** (discover, execute, security, performance, full)
- **Comprehensive logging** for audit and debugging
- **Conditional execution** based on value thresholds

### ⚙️ Configuration (`config.yaml`)
- **Adaptive scoring weights** for advanced repositories
- **Execution thresholds** and quality gates
- **Tool integration** configuration
- **Maturity tracking** and progression

## Quick Start

### 1. Run Value Discovery
```bash
# Discover and prioritize all value opportunities
python .terragon/value-discovery.py

# Check generated backlog
cat BACKLOG.md
```

### 2. Execute Highest-Value Work
```bash
# Execute the next best value item
python .terragon/autonomous-executor.py
```

### 3. Continuous Autonomous Operation
```bash
# Run full autonomous cycle
./.terragon/schedule.sh full

# Run specific cycles
./.terragon/schedule.sh discover  # Value discovery only
./.terragon/schedule.sh security  # Security scan only
./.terragon/schedule.sh execute   # Execution only
```

## Scoring Algorithm

The system uses a hybrid scoring approach optimized for advanced repositories:

### WSJF (Weighted Shortest Job First)
```
CostOfDelay = UserBusinessValue + TimeCriticality + RiskReduction + OpportunityEnablement
WSJF = CostOfDelay / JobSize
```

### ICE (Impact, Confidence, Ease)
```
ICE = Impact × Confidence × Ease
```

### Technical Debt Scoring
```
TechnicalDebtScore = (DebtImpact + DebtInterest) × HotspotMultiplier
```

### Composite Score (Advanced Repository Weights)
```
CompositeScore = 0.5×WSJF + 0.1×ICE + 0.3×TechnicalDebt + 0.1×Operational

Boosts:
- Security vulnerabilities: 2.0×
- Compliance issues: 1.8×
```

## Value Discovery Sources

- **Git History Analysis**: TODO, FIXME, HACK, DEPRECATED markers
- **Static Analysis**: Ruff, MyPy, Bandit findings
- **Security Scanning**: Vulnerability databases, dependency audits
- **Performance Analysis**: Large files, complexity hotspots
- **Documentation Gaps**: Missing docstrings, outdated docs
- **Dependency Updates**: Outdated packages, security patches

## Execution Categories

### Technical Debt
- Resolve TODO/FIXME markers
- Refactor legacy code patterns
- Improve code documentation
- Address complexity hotspots

### Security
- Patch vulnerabilities
- Update insecure dependencies
- Implement security best practices
- Add security testing

### Code Quality
- Apply automated formatting
- Fix linting violations
- Improve type annotations
- Enhance error handling

### Performance
- Optimize slow code paths
- Reduce memory usage
- Improve algorithm efficiency
- Add performance monitoring

### Documentation
- Add missing docstrings
- Update outdated documentation
- Improve API documentation
- Create usage examples

## Continuous Learning

The system continuously improves through:

- **Outcome tracking** for all executed items
- **Accuracy measurement** of score predictions
- **Velocity optimization** through process refinement
- **Pattern recognition** for similar work types
- **Adaptive weight adjustment** based on results

## Integration with CI/CD

### GitHub Actions Integration
```yaml
# .github/workflows/autonomous-sdlc.yml
name: Autonomous SDLC
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  push:
    branches: [ main ]

jobs:
  autonomous-value:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Autonomous SDLC
        run: ./.terragon/schedule.sh full
```

### Merge Trigger
```bash
# Post-merge hook
#!/bin/bash
cd /path/to/repo
./.terragon/schedule.sh discover
```

## Monitoring and Metrics

### Key Performance Indicators
- **Value Delivered**: Composite score improvements per cycle
- **Cycle Time**: Average time from discovery to PR merge
- **Success Rate**: Percentage of successful autonomous executions
- **Quality Impact**: Test coverage and code quality improvements

### Tracking Files
- `.terragon/value-metrics.json` - Current discovery metrics
- `.terragon/execution-history.json` - Historical execution data
- `.terragon/scheduler.log` - Continuous operation logs
- `BACKLOG.md` - Current prioritized backlog

## Advanced Configuration

### Custom Scoring Weights
```yaml
# config.yaml
scoring:
  weights:
    advanced:
      wsjf: 0.6        # Increase business value focus  
      technicalDebt: 0.2  # Reduce debt focus for mature repos
      security: 0.2    # Maintain security emphasis
```

### Execution Thresholds
```yaml
execution:
  minScore: 20         # Higher threshold for advanced repos
  maxRisk: 0.6         # Lower risk tolerance
  testRequirements:
    minCoverage: 90    # Higher quality standards
```

## Troubleshooting

### Common Issues

1. **No items discovered**
   - Check if discovery tools (rg, ruff) are installed
   - Verify repository has discoverable patterns
   - Lower minScore threshold

2. **Execution failures**
   - Check git configuration and permissions
   - Verify test suite is functional
   - Review validation requirements

3. **PR creation failures**
   - Ensure `gh` CLI is configured
   - Check repository permissions
   - Verify branch protection rules

### Debug Mode
```bash
# Enable detailed logging
export TERRAGON_DEBUG=1
python .terragon/value-discovery.py
```

## Contributing

To extend the autonomous system:

1. **Add new discovery sources** in `ValueDiscoveryEngine.discover_value_items()`
2. **Implement execution handlers** in `AutonomousExecutor.execute_item()`
3. **Enhance scoring algorithms** in `calculate_composite_score()`
4. **Add validation rules** in `_run_validation()`

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Value Discovery │───▶│  Scoring Engine  │───▶│  Work Selection │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Git History   │    │   WSJF + ICE +   │    │  Autonomous     │
│ Static Analysis │    │ Technical Debt   │    │   Executor      │
│Security Scanning│    │    Composite     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │    Backlog.md    │    │  Pull Request   │
                       │ Value Metrics    │    │   + Metrics     │
                       └──────────────────┘    └─────────────────┘
```

This autonomous system transforms your repository into a self-improving platform that continuously discovers, prioritizes, and executes the highest-value work, ensuring optimal SDLC maturity and sustained development velocity.