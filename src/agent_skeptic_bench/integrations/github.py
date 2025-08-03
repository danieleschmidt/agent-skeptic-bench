"""GitHub integration for Agent Skeptic Bench."""

import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import httpx


logger = logging.getLogger(__name__)


@dataclass
class GitHubWebhookEvent:
    """Represents a GitHub webhook event."""
    
    event_type: str
    action: str
    repository: str
    sender: str
    payload: Dict[str, Any]


class GitHubClient:
    """Client for interacting with GitHub API."""
    
    def __init__(self, token: Optional[str] = None, base_url: str = "https://api.github.com"):
        """Initialize GitHub client.
        
        Args:
            token: GitHub personal access token
            base_url: GitHub API base URL
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = base_url
        self.session = httpx.AsyncClient(
            headers=self._get_headers(),
            timeout=30.0
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Agent-Skeptic-Bench/1.0"
        }
        
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        
        return headers
    
    async def get_repository(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Get repository information."""
        try:
            response = await self.session.get(f"{self.base_url}/repos/{owner}/{repo}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get repository {owner}/{repo}: {e}")
            return None
    
    async def create_issue(self, owner: str, repo: str, title: str, body: str, 
                          labels: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Create a new issue."""
        try:
            payload = {
                "title": title,
                "body": body
            }
            
            if labels:
                payload["labels"] = labels
            
            response = await self.session.post(
                f"{self.base_url}/repos/{owner}/{repo}/issues",
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to create issue in {owner}/{repo}: {e}")
            return None
    
    async def create_pull_request(self, owner: str, repo: str, title: str, 
                                 head: str, base: str, body: str = "") -> Optional[Dict[str, Any]]:
        """Create a new pull request."""
        try:
            payload = {
                "title": title,
                "head": head,
                "base": base,
                "body": body
            }
            
            response = await self.session.post(
                f"{self.base_url}/repos/{owner}/{repo}/pulls",
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to create PR in {owner}/{repo}: {e}")
            return None
    
    async def add_comment(self, owner: str, repo: str, issue_number: int, 
                         body: str) -> Optional[Dict[str, Any]]:
        """Add a comment to an issue or PR."""
        try:
            payload = {"body": body}
            
            response = await self.session.post(
                f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments",
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to add comment to {owner}/{repo}#{issue_number}: {e}")
            return None
    
    async def get_workflow_runs(self, owner: str, repo: str, 
                               workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get workflow runs for a repository."""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/actions/runs"
            if workflow_id:
                url = f"{self.base_url}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs"
            
            response = await self.session.get(url)
            response.raise_for_status()
            return response.json().get("workflow_runs", [])
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to get workflow runs for {owner}/{repo}: {e}")
            return []
    
    async def create_release(self, owner: str, repo: str, tag_name: str, 
                           name: str, body: str, draft: bool = False,
                           prerelease: bool = False) -> Optional[Dict[str, Any]]:
        """Create a new release."""
        try:
            payload = {
                "tag_name": tag_name,
                "name": name,
                "body": body,
                "draft": draft,
                "prerelease": prerelease
            }
            
            response = await self.session.post(
                f"{self.base_url}/repos/{owner}/{repo}/releases",
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to create release in {owner}/{repo}: {e}")
            return None
    
    async def close(self) -> None:
        """Close the HTTP session."""
        await self.session.aclose()


class GitHubIntegration:
    """Main GitHub integration class."""
    
    def __init__(self, token: Optional[str] = None, repository: Optional[str] = None):
        """Initialize GitHub integration.
        
        Args:
            token: GitHub personal access token
            repository: Repository in format 'owner/repo'
        """
        self.client = GitHubClient(token)
        self.repository = repository or os.getenv("GITHUB_REPO")
        
        if self.repository:
            self.owner, self.repo = self.repository.split("/", 1)
        else:
            self.owner = None
            self.repo = None
    
    async def report_evaluation_results(self, session_id: str, results: Dict[str, Any]) -> bool:
        """Report evaluation results as a GitHub issue."""
        if not self.owner or not self.repo:
            logger.warning("GitHub repository not configured")
            return False
        
        # Create issue title and body
        title = f"Evaluation Results: Session {session_id}"
        
        body = f"""# Evaluation Results Report

**Session ID**: {session_id}
**Timestamp**: {results.get('timestamp', 'Unknown')}

## Summary
- **Total Scenarios**: {results.get('total_scenarios', 0)}
- **Passed Scenarios**: {results.get('passed_scenarios', 0)}
- **Pass Rate**: {results.get('pass_rate', 0):.1%}
- **Overall Score**: {results.get('overall_score', 0):.3f}

## Metrics
- **Skepticism Calibration**: {results.get('skepticism_calibration', 0):.3f}
- **Evidence Standard Score**: {results.get('evidence_standard_score', 0):.3f}
- **Red Flag Detection**: {results.get('red_flag_detection', 0):.3f}
- **Reasoning Quality**: {results.get('reasoning_quality', 0):.3f}

## Agent Configuration
- **Model**: {results.get('agent_model', 'Unknown')}
- **Provider**: {results.get('agent_provider', 'Unknown')}

---
*Automatically generated by Agent Skeptic Bench*
"""
        
        # Create the issue
        issue = await self.client.create_issue(
            self.owner, self.repo, title, body,
            labels=["evaluation-results", "automated"]
        )
        
        if issue:
            logger.info(f"Created GitHub issue #{issue['number']} for evaluation results")
            return True
        
        return False
    
    async def create_benchmark_pr(self, branch_name: str, title: str, 
                                 description: str) -> Optional[str]:
        """Create a pull request for benchmark updates."""
        if not self.owner or not self.repo:
            logger.warning("GitHub repository not configured")
            return None
        
        pr = await self.client.create_pull_request(
            self.owner, self.repo, title, branch_name, "main", description
        )
        
        if pr:
            logger.info(f"Created PR #{pr['number']}: {title}")
            return pr["html_url"]
        
        return None
    
    async def notify_benchmark_completion(self, session_id: str, 
                                        results: Dict[str, Any]) -> bool:
        """Notify about benchmark completion via GitHub comment."""
        # This would typically comment on a related issue or PR
        # For now, we'll create a new issue
        return await self.report_evaluation_results(session_id, results)
    
    def parse_webhook(self, payload: Dict[str, Any], event_type: str) -> GitHubWebhookEvent:
        """Parse GitHub webhook payload."""
        return GitHubWebhookEvent(
            event_type=event_type,
            action=payload.get("action", ""),
            repository=payload.get("repository", {}).get("full_name", ""),
            sender=payload.get("sender", {}).get("login", ""),
            payload=payload
        )
    
    async def handle_webhook(self, event: GitHubWebhookEvent) -> bool:
        """Handle GitHub webhook event."""
        logger.info(f"Received GitHub webhook: {event.event_type}.{event.action}")
        
        try:
            if event.event_type == "push":
                return await self._handle_push_event(event)
            elif event.event_type == "pull_request":
                return await self._handle_pr_event(event)
            elif event.event_type == "issue":
                return await self._handle_issue_event(event)
            else:
                logger.debug(f"Unhandled webhook event: {event.event_type}")
                return True
                
        except Exception as e:
            logger.error(f"Error handling webhook event: {e}")
            return False
    
    async def _handle_push_event(self, event: GitHubWebhookEvent) -> bool:
        """Handle push events."""
        # Check if push is to main branch
        ref = event.payload.get("ref", "")
        if ref == "refs/heads/main":
            logger.info("Push to main branch detected - could trigger benchmark run")
            # Here we could trigger automatic benchmark evaluation
        
        return True
    
    async def _handle_pr_event(self, event: GitHubWebhookEvent) -> bool:
        """Handle pull request events."""
        action = event.action
        pr_number = event.payload.get("number")
        
        if action == "opened":
            logger.info(f"New PR #{pr_number} opened")
            # Could add automatic benchmark comment
        elif action == "closed" and event.payload.get("pull_request", {}).get("merged"):
            logger.info(f"PR #{pr_number} merged")
            # Could trigger benchmark run
        
        return True
    
    async def _handle_issue_event(self, event: GitHubWebhookEvent) -> bool:
        """Handle issue events."""
        action = event.action
        issue_number = event.payload.get("number")
        
        if action == "opened":
            logger.info(f"New issue #{issue_number} opened")
            # Could check if it's a benchmark request
        
        return True
    
    async def close(self) -> None:
        """Close the GitHub client."""
        await self.client.close()


# GitHub Actions integration
class GitHubActionsReporter:
    """Reporter for GitHub Actions workflows."""
    
    @staticmethod
    def set_output(name: str, value: str) -> None:
        """Set GitHub Actions output variable."""
        if os.getenv("GITHUB_ACTIONS"):
            print(f"::set-output name={name}::{value}")
    
    @staticmethod
    def create_summary(title: str, content: str) -> None:
        """Create GitHub Actions job summary."""
        if os.getenv("GITHUB_ACTIONS"):
            summary_file = os.getenv("GITHUB_STEP_SUMMARY")
            if summary_file:
                with open(summary_file, "a") as f:
                    f.write(f"## {title}\n\n")
                    f.write(content)
                    f.write("\n\n")
    
    @staticmethod
    def add_annotation(level: str, message: str, title: str = "", 
                      file: str = "", line: int = 0) -> None:
        """Add GitHub Actions annotation."""
        if os.getenv("GITHUB_ACTIONS"):
            annotation = f"::{level}"
            
            if title:
                annotation += f" title={title}"
            if file:
                annotation += f" file={file}"
            if line:
                annotation += f" line={line}"
            
            annotation += f"::{message}"
            print(annotation)