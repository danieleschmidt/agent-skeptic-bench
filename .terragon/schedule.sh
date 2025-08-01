#!/bin/bash
# Terragon Autonomous SDLC Scheduler
# Continuous value discovery and execution

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a .terragon/scheduler.log
}

run_value_discovery() {
    log "🔍 Running value discovery..."
    python .terragon/value-discovery.py
    if [ $? -eq 0 ]; then
        log "✅ Value discovery completed"
    else
        log "❌ Value discovery failed"
    fi
}

run_autonomous_execution() {
    log "⚡ Running autonomous execution..."
    python .terragon/autonomous-executor.py
    if [ $? -eq 0 ]; then
        log "✅ Autonomous execution completed"
    else
        log "❌ Autonomous execution failed"
    fi
}

run_security_scan() {
    log "🔒 Running security scan..."
    
    # Run safety check if available
    if command -v safety &> /dev/null; then
        safety check --output json > .terragon/security-scan.json 2>/dev/null
        log "✅ Security scan completed"
    else
        log "⚠️  Safety not available - skipping security scan"
    fi
}

run_performance_analysis() {
    log "📊 Running performance analysis..."
    
    # Basic file size analysis
    find src/ -name "*.py" -exec wc -l {} + | sort -nr | head -5 > .terragon/large-files.txt
    log "✅ Performance analysis completed"
}

# Main execution based on schedule type
case "${1:-discover}" in
    "discover")
        log "🚀 Starting value discovery cycle"
        run_value_discovery
        ;;
    "execute")
        log "🚀 Starting autonomous execution cycle"
        run_autonomous_execution
        ;;
    "security")
        log "🚀 Starting security scan cycle"
        run_security_scan
        ;;
    "performance")
        log "🚀 Starting performance analysis cycle"
        run_performance_analysis
        ;;
    "full")
        log "🚀 Starting full autonomous cycle"
        run_value_discovery
        run_security_scan
        run_performance_analysis
        
        # Only execute if high-value items found
        if [ -f "BACKLOG.md" ] && grep -q "Score.*[5-9][0-9]" BACKLOG.md; then
            run_autonomous_execution
        else
            log "📋 No high-value items found - skipping execution"
        fi
        ;;
    *)
        echo "Usage: $0 [discover|execute|security|performance|full]"
        echo "  discover    - Run value discovery only"
        echo "  execute     - Run autonomous execution only"
        echo "  security    - Run security scan only"
        echo "  performance - Run performance analysis only"
        echo "  full        - Run complete autonomous cycle"
        exit 1
        ;;
esac

log "🏁 Cycle completed"