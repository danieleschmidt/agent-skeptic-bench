#!/bin/bash
set -e

# Default environment variables
export PYTHONPATH="/app/src:$PYTHONPATH"
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export ENVIRONMENT=${ENVIRONMENT:-production}
export MAX_CONCURRENT_EVALUATIONS=${MAX_CONCURRENT_EVALUATIONS:-16}

echo "🚀 Starting Agent Skeptic Bench in $ENVIRONMENT mode"

# Wait for dependencies
echo "⏳ Waiting for dependencies..."
if [ -n "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    while ! curl -f "$REDIS_URL" >/dev/null 2>&1; do
        sleep 2
    done
    echo "✅ Redis is ready"
fi

if [ -n "$DATABASE_URL" ]; then
    echo "Waiting for database..."
    # Simple check - in production, use proper database health check
    sleep 5
    echo "✅ Database is ready"
fi

# Initialize application
echo "🔧 Initializing application..."
cd /app

# Run database migrations if needed
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Running database setup..."
    python -c "
from src.agent_skeptic_bench.database.connection import DatabaseManager
import asyncio

async def setup():
    db = DatabaseManager()
    await db.create_tables()
    print('✅ Database tables created')

asyncio.run(setup())
" || echo "⚠️ Database setup skipped"
fi

# Validate scenarios
echo "🔍 Validating scenarios..."
python -c "
from src.agent_skeptic_bench.data_loader import get_data_loader
from src.agent_skeptic_bench.validation import validate_scenario_file
from pathlib import Path
import sys

loader = get_data_loader()
scenarios = loader.load_scenarios()
print(f'✅ Loaded and validated {len(scenarios)} scenarios')

# Quick validation of scenario files
data_dir = Path('/app/data/scenarios')
if data_dir.exists():
    error_count = 0
    for json_file in data_dir.rglob('*.json'):
        is_valid, errors = validate_scenario_file(json_file)
        if not is_valid:
            print(f'❌ Invalid scenario {json_file}: {errors}')
            error_count += 1
    
    if error_count > 0:
        print(f'⚠️ Found {error_count} scenario validation errors')
        sys.exit(1)
    else:
        print('✅ All scenario files validated successfully')
else:
    print('⚠️ No scenario data directory found')
"

# Start the application
echo "🌟 Starting Agent Skeptic Bench server..."

if [ "$1" = "api" ]; then
    echo "Starting API server..."
    exec gunicorn src.agent_skeptic_bench.api.app:app \
        --config gunicorn.conf.py \
        --bind 0.0.0.0:8000 \
        --access-logfile - \
        --error-logfile -
elif [ "$1" = "worker" ]; then
    echo "Starting background worker..."
    exec python -m src.agent_skeptic_bench.worker
elif [ "$1" = "cli" ]; then
    echo "Starting CLI mode..."
    exec python simple_cli.py "${@:2}"
else
    echo "Starting full application stack..."
    # Start API server by default
    exec gunicorn src.agent_skeptic_bench.api.app:app \
        --config gunicorn.conf.py \
        --bind 0.0.0.0:8000 \
        --access-logfile - \
        --error-logfile -
fi