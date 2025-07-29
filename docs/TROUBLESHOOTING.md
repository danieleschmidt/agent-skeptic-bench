# Troubleshooting Guide

Comprehensive troubleshooting for common issues in Agent Skeptic Bench.

## Common Issues

### Installation and Setup

#### Issue: Package Installation Fails

**Symptoms:**
```bash
ERROR: Could not install packages due to an EnvironmentError
pip: error: Microsoft Visual C++ 14.0 is required
```

**Solutions:**

1. **Windows Users:**
   ```bash
   # Install Microsoft C++ Build Tools
   # Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   
   # Alternative: Use conda instead of pip
   conda create -n skeptic-bench python=3.11
   conda activate skeptic-bench
   conda install -c conda-forge agent-skeptic-bench
   ```

2. **macOS Users:**
   ```bash
   # Install Xcode command line tools
   xcode-select --install
   
   # Update pip and setuptools
   pip install --upgrade pip setuptools wheel
   ```

3. **Linux Users:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install build-essential python3-dev
   
   # CentOS/RHEL
   sudo yum groupinstall "Development Tools"
   sudo yum install python3-devel
   ```

#### Issue: Import Errors

**Symptoms:**
```python
ImportError: No module named 'agent_skeptic_bench'
ModuleNotFoundError: No module named 'openai'
```

**Solutions:**

1. **Verify Installation:**
   ```bash
   pip show agent-skeptic-bench
   pip list | grep agent-skeptic-bench
   ```

2. **Check Python Path:**
   ```python
   import sys
   print(sys.path)
   print(sys.executable)
   ```

3. **Reinstall in Development Mode:**
   ```bash
   pip uninstall agent-skeptic-bench
   pip install -e ".[dev]"
   ```

### API and Authentication

#### Issue: API Key Authentication Fails

**Symptoms:**
```
AuthenticationError: Invalid API key provided
RateLimitError: You exceeded your current quota
```

**Solutions:**

1. **Verify API Key Format:**
   ```bash
   # OpenAI keys start with 'sk-'
   echo $OPENAI_API_KEY | grep '^sk-'
   
   # Anthropic keys are longer alphanumeric strings
   echo $ANTHROPIC_API_KEY | wc -c
   ```

2. **Check API Key Permissions:**
   ```python
   import openai
   
   # Test API key validity
   try:
       client = openai.OpenAI(api_key="your-key-here")
       models = client.models.list()
       print("API key is valid")
   except Exception as e:
       print(f"API key error: {e}")
   ```

3. **Environment Variable Issues:**
   ```bash
   # Check if environment variables are set
   env | grep API_KEY
   
   # Set temporarily for testing
   export OPENAI_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
   ```

4. **Rate Limit Handling:**
   ```python
   # Implement exponential backoff
   import time
   import random
   
   def retry_with_backoff(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return func()
           except RateLimitError:
               if attempt == max_retries - 1:
                   raise
               wait_time = (2 ** attempt) + random.uniform(0, 1)
               time.sleep(wait_time)
   ```

#### Issue: Network Connectivity Problems

**Symptoms:**
```
ConnectionError: HTTPSConnectionPool
SSLError: certificate verify failed
TimeoutError: Request timed out
```

**Solutions:**

1. **Corporate Firewall/Proxy:**
   ```bash
   # Set proxy environment variables
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   
   # Or configure in requests
   proxies = {
       'http': 'http://proxy.company.com:8080',
       'https': 'http://proxy.company.com:8080'
   }
   ```

2. **SSL Certificate Issues:**
   ```python
   import ssl
   import certifi
   
   # Use system certificates
   ssl_context = ssl.create_default_context(cafile=certifi.where())
   ```

3. **Timeout Configuration:**
   ```python
   # Increase timeout values
   client = openai.OpenAI(
       timeout=60,  # 60 second timeout
       max_retries=3
   )
   ```

### Performance Issues

#### Issue: Slow Evaluation Performance

**Symptoms:**
- Evaluations taking longer than expected
- High CPU/memory usage
- Timeouts during batch processing

**Diagnosis:**

1. **Profile Performance:**
   ```python
   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   
   # Run your evaluation code here
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)
   ```

2. **Monitor Resource Usage:**
   ```bash
   # Monitor in real-time
   htop
   
   # Log resource usage
   python -c "
   import psutil
   import time
   
   while True:
       cpu = psutil.cpu_percent()
       memory = psutil.virtual_memory().percent
       print(f'CPU: {cpu}%, Memory: {memory}%')
       time.sleep(5)
   "
   ```

**Solutions:**

1. **Optimize Batch Size:**
   ```python
   # Reduce batch size to prevent memory issues
   batch_size = 10  # Instead of 100
   
   for i in range(0, len(scenarios), batch_size):
       batch = scenarios[i:i + batch_size]
       results = evaluate_batch(batch)
   ```

2. **Enable Caching:**
   ```python
   from agent_skeptic_bench import cache_manager
   
   # Cache scenario results
   @cache_manager.cached('scenario_result')
   def evaluate_scenario(scenario_id, model):
       # Expensive evaluation logic
       pass
   ```

3. **Use Async Processing:**
   ```python
   import asyncio
   
   async def evaluate_scenarios_async(scenarios):
       tasks = [evaluate_single_async(s) for s in scenarios]
       results = await asyncio.gather(*tasks)
       return results
   ```

#### Issue: Memory Leaks

**Symptoms:**
- Memory usage continuously increasing
- Out of memory errors during long runs
- System becoming unresponsive

**Diagnosis:**

1. **Memory Profiling:**
   ```python
   from memory_profiler import profile
   
   @profile
   def memory_intensive_function():
       # Your function here
       pass
   ```

2. **Track Object Growth:**
   ```python
   import gc
   import objgraph
   
   # Show most common types
   objgraph.show_most_common_types()
   
   # Track object growth
   objgraph.show_growth()
   ```

**Solutions:**

1. **Explicit Memory Management:**
   ```python
   # Clear variables when done
   large_data = process_large_dataset()
   # Use the data
   del large_data
   gc.collect()  # Force garbage collection
   ```

2. **Generator Usage:**
   ```python
   # Instead of loading all scenarios at once
   def load_all_scenarios():
       return [load_scenario(i) for i in range(1000)]  # Memory intensive
   
   # Use generator
   def scenario_generator():
       for i in range(1000):
           yield load_scenario(i)  # Memory efficient
   ```

### Database and Storage Issues

#### Issue: Database Connection Errors

**Symptoms:**
```
OperationalError: could not connect to server
InterfaceError: connection already closed
DatabaseError: relation does not exist
```

**Solutions:**

1. **Connection Pool Configuration:**
   ```python
   from sqlalchemy import create_engine
   
   engine = create_engine(
       database_url,
       pool_size=10,
       max_overflow=20,
       pool_pre_ping=True,  # Verify connections
       pool_recycle=3600    # Recycle hourly
   )
   ```

2. **Database Migration:**
   ```bash
   # Run database migrations
   alembic upgrade head
   
   # Or create tables manually
   python -c "from agent_skeptic_bench.database import create_tables; create_tables()"
   ```

3. **Connection Retry Logic:**
   ```python
   from sqlalchemy.exc import DisconnectionError
   import time
   
   def robust_db_operation(operation, max_retries=3):
       for attempt in range(max_retries):
           try:
               return operation()
           except DisconnectionError:
               if attempt == max_retries - 1:
                   raise
               time.sleep(2 ** attempt)
   ```

#### Issue: Disk Space Problems

**Symptoms:**
```
OSError: [Errno 28] No space left on device
PermissionError: [Errno 13] Permission denied
```

**Solutions:**

1. **Check Disk Usage:**
   ```bash
   # Check overall disk usage
   df -h
   
   # Find large files
   du -h --max-depth=1 | sort -hr
   
   # Find files larger than 100MB
   find . -size +100M -type f -exec ls -lh {} \;
   ```

2. **Clean Up Temporary Files:**
   ```bash
   # Clean Python cache
   find . -type d -name "__pycache__" -exec rm -r {} +
   find . -name "*.pyc" -delete
   
   # Clean logs older than 7 days
   find /app/logs -name "*.log" -mtime +7 -delete
   ```

3. **Implement Log Rotation:**
   ```python
   import logging.handlers
   
   # Rotating file handler
   handler = logging.handlers.RotatingFileHandler(
       'app.log',
       maxBytes=10*1024*1024,  # 10MB
       backupCount=5
   )
   ```

### Docker and Container Issues

#### Issue: Container Build Failures

**Symptoms:**
```
ERROR: failed to solve: process "/bin/sh -c pip install" did not complete successfully
failed to compute cache key: failed to walk /var/lib/docker/tmp
```

**Solutions:**

1. **Clear Docker Cache:**
   ```bash
   # Clean Docker system
   docker system prune -a
   
   # Remove unused images
   docker image prune -a
   
   # Clear build cache
   docker builder prune
   ```

2. **Multi-stage Build Issues:**
   ```dockerfile
   # Ensure proper stage naming
   FROM python:3.11-slim as builder
   # ... build stage
   
   FROM python:3.11-slim as production
   COPY --from=builder /opt/venv /opt/venv
   ```

3. **Permission Issues:**
   ```dockerfile
   # Create user with specific UID
   RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser
   
   # Set ownership correctly
   COPY --chown=appuser:appgroup src/ /app/src/
   ```

#### Issue: Container Runtime Problems

**Symptoms:**
- Container exits immediately
- Port binding failures
- Volume mount issues

**Solutions:**

1. **Debug Container Startup:**
   ```bash
   # Run interactively
   docker run -it --entrypoint /bin/bash your-image
   
   # Check logs
   docker logs container-name
   
   # Inspect container
   docker inspect container-name
   ```

2. **Port Conflicts:**
   ```bash
   # Check what's using the port
   lsof -i :8080
   netstat -tulpn | grep :8080
   
   # Use different port
   docker run -p 8081:8080 your-image
   ```

3. **Volume Permission Issues:**
   ```bash
   # Fix volume permissions
   docker run --user $(id -u):$(id -g) -v $(pwd):/app your-image
   
   # Or in Dockerfile
   RUN chown -R appuser:appgroup /app
   ```

### Testing Issues

#### Issue: Test Failures

**Symptoms:**
```
ASSERTIONERROR: Skepticism score too low
FIXTURE ERROR: Could not create test scenario
TimeoutError: Test took too long
```

**Solutions:**

1. **Mock External Dependencies:**
   ```python
   import pytest
   from unittest.mock import Mock, patch
   
   @patch('agent_skeptic_bench.openai_client')
   def test_evaluation(mock_client):
       mock_client.chat.completions.create.return_value.choices[0].message.content = "Mock response"
       # Your test code here
   ```

2. **Increase Test Timeouts:**
   ```python
   @pytest.mark.timeout(300)  # 5 minute timeout
   def test_long_running_evaluation():
       # Long running test
       pass
   ```

3. **Parameterized Tests:**
   ```python
   @pytest.mark.parametrize("model", ["gpt-4", "claude-3"])
   @pytest.mark.parametrize("category", ["factual_claims", "flawed_plans"])
   def test_all_combinations(model, category):
       # Test all model/category combinations
       pass
   ```

## Debugging Tools and Techniques

### Logging Configuration

```python
# Enhanced logging for debugging
import logging
import sys
from datetime import datetime

def setup_debug_logging():
    """Configure comprehensive debug logging."""
    
    # Create logger
    logger = logging.getLogger('agent_skeptic_bench')
    logger.setLevel(logging.DEBUG)
    
    # Console handler with detailed format
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler for persistent logs
    file_handler = logging.FileHandler(f'debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Set levels for third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return logger
```

### Debug Utilities

```python
# Debug utilities
import functools
import time
import traceback
from typing import Any, Callable

def debug_function_calls(func: Callable) -> Callable:
    """Decorator to debug function calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        
        # Log function entry
        logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Exiting {func.__name__} after {execution_time:.3f}s with result type: {type(result)}")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Exception in {func.__name__} after {execution_time:.3f}s: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    return wrapper

def debug_variable_state(**variables):
    """Log current state of variables."""
    logger = logging.getLogger(__name__)
    for name, value in variables.items():
        logger.debug(f"Variable {name}: {type(value)} = {repr(value)[:200]}")

# Usage
@debug_function_calls
def evaluate_scenario(scenario_id: str, model: str):
    debug_variable_state(scenario_id=scenario_id, model=model)
    # Function implementation
    pass
```

### Health Check Endpoints

```python
# Health check endpoints for debugging
from flask import Flask, jsonify
import psutil
import sys
import platform

app = Flask(__name__)

@app.route('/debug/health')
def debug_health():
    """Comprehensive health check for debugging."""
    try:
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'system': {
                'platform': platform.platform(),
                'python_version': sys.version,
                'architecture': platform.architecture(),
                'hostname': platform.node()
            },
            'resources': {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            'environment': {
                'has_openai_key': bool(os.getenv('OPENAI_API_KEY')),
                'has_anthropic_key': bool(os.getenv('ANTHROPIC_API_KEY')),
                'python_path': sys.path[:3],  # First 3 entries
                'working_directory': os.getcwd()
            },
            'dependencies': {
                'installed_packages': get_installed_packages()[:10]  # Top 10
            }
        }
        
        return jsonify(health_data)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

def get_installed_packages():
    """Get list of installed packages."""
    try:
        import pkg_resources
        return [f"{pkg.project_name}=={pkg.version}" for pkg in pkg_resources.working_set]
    except ImportError:
        return []

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
```

## Getting Help

### Community Resources

1. **GitHub Issues**: https://github.com/yourusername/agent-skeptic-bench/issues
2. **Discussions**: https://github.com/yourusername/agent-skeptic-bench/discussions
3. **Discord**: https://discord.gg/skeptic-bench
4. **Documentation**: https://agent-skeptic-bench.org/docs

### Reporting Issues

When reporting issues, please include:

1. **Environment Information:**
   ```bash
   # System information
   python --version
   pip show agent-skeptic-bench
   
   # Operating system
   uname -a  # Linux/macOS
   systeminfo  # Windows
   ```

2. **Error Details:**
   - Full error message and traceback
   - Steps to reproduce
   - Expected vs actual behavior
   - Relevant configuration files

3. **Debug Logs:**
   ```python
   # Enable debug logging before running
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Emergency Contacts

For critical production issues:
- **Email**: emergency@skeptic-bench.org
- **Slack**: #urgent-support (for enterprise customers)
- **Phone**: Available for enterprise support customers

---

*This troubleshooting guide is regularly updated. For the latest version, visit: https://agent-skeptic-bench.org/docs/troubleshooting*
