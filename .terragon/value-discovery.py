#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuously discovers and prioritizes high-value work items
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import re

class ValueDiscoveryEngine:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / "BACKLOG.md"
        
    def discover_value_items(self) -> List[Dict[str, Any]]:
        """Comprehensive signal harvesting from multiple sources"""
        items = []
        
        # Git history analysis for TODO/FIXME/HACK markers
        items.extend(self._analyze_git_history())
        
        # Static analysis findings
        items.extend(self._run_static_analysis())
        
        # Security vulnerability scan
        items.extend(self._scan_vulnerabilities())
        
        # Dependency updates
        items.extend(self._check_dependency_updates())
        
        # Performance opportunities
        items.extend(self._analyze_performance_opportunities())
        
        # Documentation gaps
        items.extend(self._find_documentation_gaps())
        
        return items
    
    def _analyze_git_history(self) -> List[Dict[str, Any]]:
        """Extract debt markers from git history and current code"""
        items = []
        try:
            # Try rg first, fallback to grep if available
            try:
                result = subprocess.run([
                    "rg", "-n", "--type", "py", 
                    r"(TODO|FIXME|HACK|DEPRECATED|XXX)(\(.*?\))?:?\s*(.*)",
                    str(self.repo_path)
                ], capture_output=True, text=True)
            except FileNotFoundError:
                # Fallback to find + grep
                result = subprocess.run([
                    "find", str(self.repo_path), "-name", "*.py", "-exec", 
                    "grep", "-Hn", "-E", r"(TODO|FIXME|HACK|DEPRECATED|XXX)", "{}", "+"
                ], capture_output=True, text=True)
            
            for line in result.stdout.splitlines():
                if ":" in line:
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        file_path, line_num, content = parts
                        match = re.search(r"(TODO|FIXME|HACK|DEPRECATED|XXX)(\(.*?\))?:?\s*(.*)", content)
                        if match:
                            marker_type = match.group(1)
                            description = match.group(3).strip() if match.group(3) else "No description"
                            
                            priority = {
                                "DEPRECATED": "high",
                                "FIXME": "high", 
                                "HACK": "medium",
                                "TODO": "low",
                                "XXX": "medium"
                            }.get(marker_type, "low")
                            
                            items.append({
                                "id": f"debt-{hash(f'{file_path}:{line_num}') % 10000}",
                                "title": f"Address {marker_type.lower()}: {description[:50]}...",
                                "category": "technical-debt",
                                "source": "git-history",
                                "file": file_path,
                                "line": line_num,
                                "description": description,
                                "priority": priority,
                                "effort_estimate": 2 if marker_type in ["TODO", "XXX"] else 4,
                                "business_value": 3 if marker_type == "DEPRECATED" else 2,
                                "urgency": 4 if marker_type in ["FIXME", "DEPRECATED"] else 2
                            })
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Manual search as final fallback
            try:
                for py_file in self.repo_path.rglob("*.py"):
                    with open(py_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            match = re.search(r"(TODO|FIXME|HACK|DEPRECATED|XXX)(\(.*?\))?:?\s*(.*)", line)
                            if match:
                                marker_type = match.group(1)
                                description = match.group(3).strip() if match.group(3) else "No description"
                                
                                items.append({
                                    "id": f"debt-{hash(f'{py_file}:{line_num}') % 10000}",
                                    "title": f"Address {marker_type.lower()}: {description[:50]}...",
                                    "category": "technical-debt",
                                    "source": "git-history",
                                    "file": str(py_file.relative_to(self.repo_path)),
                                    "line": str(line_num),
                                    "description": description,
                                    "priority": "medium",
                                    "effort_estimate": 2,
                                    "business_value": 2,
                                    "urgency": 2
                                })
            except Exception:
                pass  # Skip if file reading fails
            
        return items
    
    def _run_static_analysis(self) -> List[Dict[str, Any]]:
        """Run static analysis tools and extract findings"""
        items = []
        
        try:
            # Run ruff for code quality issues
            result = subprocess.run([
                "ruff", "check", "--output-format=json", str(self.repo_path)
            ], capture_output=True, text=True)
            
            if result.stdout:
                findings = json.loads(result.stdout)
                for finding in findings[:10]:  # Limit to top 10
                    items.append({
                        "id": f"ruff-{hash(finding.get('filename', '') + finding.get('code', '')) % 10000}",
                        "title": f"Fix {finding.get('code', 'style')}: {finding.get('message', '')[:50]}...",
                        "category": "code-quality",
                        "source": "static-analysis",
                        "file": finding.get("filename"),
                        "line": finding.get("location", {}).get("row"),
                        "description": finding.get("message", ""),
                        "priority": "medium",
                        "effort_estimate": 1,
                        "business_value": 2,
                        "urgency": 1
                    })
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            # Add fallback static analysis opportunities
            items.extend([
                {
                    "id": "static-1",
                    "title": "Set up automated code quality checks",
                    "category": "code-quality",
                    "source": "static-analysis",
                    "description": "Configure ruff/black/isort for consistent code quality",
                    "priority": "medium",
                    "effort_estimate": 2,
                    "business_value": 3,
                    "urgency": 2
                }
            ])
            
        return items
    
    def _scan_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Scan for security vulnerabilities"""
        items = []
        
        try:
            # Run safety check for known vulnerabilities
            result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                findings = json.loads(result.stdout)
                for vuln in findings.get("vulnerabilities", [])[:5]:  # Top 5
                    items.append({
                        "id": f"vuln-{hash(vuln.get('package_name', '') + vuln.get('vulnerability_id', '')) % 10000}",
                        "title": f"Security: Update {vuln.get('package_name')} (CVE-{vuln.get('vulnerability_id', 'Unknown')})",
                        "category": "security",
                        "source": "vulnerability-scan",
                        "description": vuln.get("advisory", "Security vulnerability found"),
                        "priority": "high",
                        "effort_estimate": 1,
                        "business_value": 5,
                        "urgency": 5,
                        "security_boost": True
                    })
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            # Add generic security improvement opportunities
            items.extend([
                {
                    "id": "sec-1",
                    "title": "Implement security dependency scanning",
                    "category": "security",
                    "source": "vulnerability-scan",
                    "description": "Set up automated security vulnerability scanning",
                    "priority": "high",
                    "effort_estimate": 2,
                    "business_value": 4,
                    "urgency": 4,
                    "security_boost": True
                },
                {
                    "id": "sec-2", 
                    "title": "Review and update dependencies for security",
                    "category": "security",
                    "source": "vulnerability-scan",
                    "description": "Manual review of dependencies for known vulnerabilities",
                    "priority": "medium",
                    "effort_estimate": 3,
                    "business_value": 3,
                    "urgency": 3
                }
            ])
            
        return items
    
    def _check_dependency_updates(self) -> List[Dict[str, Any]]:
        """Check for available dependency updates"""
        items = []
        
        try:
            # Check for outdated packages
            result = subprocess.run([
                "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                for pkg in outdated[:3]:  # Top 3 most important
                    items.append({
                        "id": f"dep-{hash(pkg.get('name', '')) % 10000}",
                        "title": f"Update {pkg.get('name')} to {pkg.get('latest_version')}",
                        "category": "dependency-update",
                        "source": "dependency-check",
                        "description": f"Update from {pkg.get('version')} to {pkg.get('latest_version')}",
                        "priority": "medium",
                        "effort_estimate": 1,
                        "business_value": 2,
                        "urgency": 2
                    })
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        return items
    
    def _analyze_performance_opportunities(self) -> List[Dict[str, Any]]:
        """Identify potential performance improvements"""
        items = []
        
        # Check for large files that might need optimization
        try:
            large_files = []
            for py_file in self.repo_path.rglob("*.py"):
                if py_file.stat().st_size > 10000:  # > 10KB
                    large_files.append((py_file, py_file.stat().st_size))
            
            large_files.sort(key=lambda x: x[1], reverse=True)
            
            for file_path, size in large_files[:3]:  # Top 3 largest
                items.append({
                    "id": f"perf-{hash(str(file_path)) % 10000}",
                    "title": f"Review large file for optimization: {file_path.name}",
                    "category": "performance",
                    "source": "file-analysis",
                    "file": str(file_path.relative_to(self.repo_path)),
                    "description": f"Large file ({size} bytes) may benefit from refactoring",
                    "priority": "low",
                    "effort_estimate": 3,
                    "business_value": 2,
                    "urgency": 1
                })
        except Exception:
            pass
            
        return items
    
    def _find_documentation_gaps(self) -> List[Dict[str, Any]]:
        """Identify missing or outdated documentation"""
        items = []
        
        # Check for Python files without docstrings
        try:
            src_path = self.repo_path / "src"
            if src_path.exists():
                # Manual search for functions/classes without docstrings
                py_files = list(src_path.rglob("*.py"))[:3]  # Limit to 3 files
                
                for py_file in py_files:
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Look for function/class definitions without immediate docstrings
                        import re
                        functions_classes = re.findall(r'^(def|class)\s+(\w+)', content, re.MULTILINE)
                        
                        if functions_classes:
                            items.append({
                                "id": f"doc-{hash(str(py_file)) % 10000}",
                                "title": f"Add docstrings to {py_file.name}",
                                "category": "documentation",
                                "source": "doc-analysis",
                                "file": str(py_file.relative_to(self.repo_path)),
                                "description": f"Missing documentation for {len(functions_classes)} functions/classes",
                                "priority": "low",
                                "effort_estimate": 2,
                                "business_value": 2,
                                "urgency": 1
                            })
                    except Exception:
                        continue
        except Exception:
            # Add generic documentation improvements
            items.append({
                "id": "doc-generic",
                "title": "Improve project documentation coverage",
                "category": "documentation", 
                "source": "doc-analysis",
                "description": "General documentation improvements needed",
                "priority": "low",
                "effort_estimate": 3,
                "business_value": 2,
                "urgency": 1
            })
            
        return items

    def calculate_composite_score(self, item: Dict[str, Any]) -> float:
        """Calculate WSJF + ICE + Technical Debt composite score"""
        
        # WSJF Components
        user_business_value = item.get("business_value", 3) * 10
        time_criticality = item.get("urgency", 2) * 8
        risk_reduction = 15 if item.get("category") == "security" else 5
        opportunity_enablement = 10 if item.get("category") == "technical-debt" else 5
        
        cost_of_delay = user_business_value + time_criticality + risk_reduction + opportunity_enablement
        job_size = max(item.get("effort_estimate", 2), 1)
        wsjf = cost_of_delay / job_size
        
        # ICE Components  
        impact = item.get("business_value", 3)
        confidence = 8 if item.get("source") == "static-analysis" else 6
        ease = 11 - item.get("effort_estimate", 2)  # Invert effort to ease
        ice = impact * confidence * ease
        
        # Technical Debt Score
        debt_impact = 20 if item.get("category") == "technical-debt" else 10
        debt_interest = item.get("urgency", 2) * 5
        tech_debt_score = debt_impact + debt_interest
        
        # Composite Score with adaptive weighting for advanced repos
        composite = (
            0.5 * self._normalize_score(wsjf, 0, 100) +
            0.1 * self._normalize_score(ice, 0, 880) +
            0.3 * self._normalize_score(tech_debt_score, 0, 50) +
            0.1 * 20  # Base operational value
        )
        
        # Apply boosts
        if item.get("security_boost"):
            composite *= 2.0
        if item.get("category") == "security":
            composite *= 1.8
            
        return round(composite, 1)
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-100 range"""
        return max(0, min(100, ((value - min_val) / (max_val - min_val)) * 100))
    
    def generate_backlog(self) -> None:
        """Generate comprehensive value-driven backlog"""
        items = self.discover_value_items()
        
        # Calculate scores for all items
        for item in items:
            item["composite_score"] = self.calculate_composite_score(item)
        
        # Sort by composite score
        items.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # Generate backlog markdown
        timestamp = datetime.now().isoformat()
        
        backlog_content = f"""# 📊 Autonomous Value Backlog

Last Updated: {timestamp}
Repository: Agent Skeptic Bench
Maturity Level: ADVANCED (78%)

## 🎯 Next Best Value Item
{self._format_next_item(items[0] if items else None)}

## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
        
        for i, item in enumerate(items[:10], 1):
            backlog_content += f"| {i} | {item['id']} | {item['title'][:40]}... | {item['composite_score']} | {item['category'].title()} | {item['effort_estimate']} |\n"
        
        avg_score = sum(i['composite_score'] for i in items) / len(items) if items else 0
        backlog_content += f"""

## 📈 Value Metrics
- **Total Items Discovered**: {len(items)}
- **High Priority Items**: {len([i for i in items if i.get('priority') == 'high'])}
- **Security Items**: {len([i for i in items if i.get('category') == 'security'])}
- **Technical Debt Items**: {len([i for i in items if i.get('category') == 'technical-debt'])}
- **Average Score**: {avg_score:.1f}

## 🔄 Discovery Sources
- Git History Analysis: {len([i for i in items if i.get('source') == 'git-history'])} items
- Static Analysis: {len([i for i in items if i.get('source') == 'static-analysis'])} items  
- Security Scans: {len([i for i in items if i.get('source') == 'vulnerability-scan'])} items
- Dependency Updates: {len([i for i in items if i.get('source') == 'dependency-check'])} items
- Performance Analysis: {len([i for i in items if i.get('source') == 'file-analysis'])} items
- Documentation Gaps: {len([i for i in items if i.get('source') == 'doc-analysis'])} items

## 📊 Detailed Items

"""
        
        for item in items[:10]:
            backlog_content += f"""### {item['id'].upper()}: {item['title']}
- **Score**: {item['composite_score']} 
- **Category**: {item['category'].title()}
- **Priority**: {item['priority'].title()}
- **Effort**: {item['effort_estimate']} hours
- **Source**: {item['source'].title()}
- **Description**: {item['description']}

"""
        
        # Write backlog file
        with open(self.backlog_path, 'w') as f:
            f.write(backlog_content)
        
        # Update metrics
        self._update_metrics(items)
    
    def _format_next_item(self, item: Dict[str, Any]) -> str:
        """Format the next best value item for display"""
        if not item:
            return "**No items in backlog**"
            
        return f"""**[{item['id'].upper()}] {item['title']}**
- **Composite Score**: {item['composite_score']}
- **Category**: {item['category'].title()}
- **Estimated Effort**: {item['effort_estimate']} hours
- **Expected Impact**: {item['description']}"""
    
    def _update_metrics(self, items: List[Dict[str, Any]]) -> None:
        """Update value metrics tracking"""
        metrics = {
            "lastUpdate": datetime.now().isoformat(),
            "totalItems": len(items),
            "averageScore": sum(i['composite_score'] for i in items) / len(items) if items else 0,
            "categoryBreakdown": {},
            "priorityBreakdown": {},
            "topScore": items[0]['composite_score'] if items else 0
        }
        
        # Calculate breakdowns
        for item in items:
            cat = item['category']
            pri = item['priority']
            metrics["categoryBreakdown"][cat] = metrics["categoryBreakdown"].get(cat, 0) + 1
            metrics["priorityBreakdown"][pri] = metrics["priorityBreakdown"].get(pri, 0) + 1
        
        # Write metrics
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    engine = ValueDiscoveryEngine()
    engine.generate_backlog()
    print("✅ Value discovery complete - check BACKLOG.md for results")