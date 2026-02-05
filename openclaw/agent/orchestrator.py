"""
Agent Orchestrator - Hierarchical Multi-Agent System
Inspired by AgentZero's delegation and hierarchy model.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.agent.orchestrator")


@dataclass
class AgentNode:
    """Represents an agent in the hierarchy."""
    id: str = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
    name: str = "Agent"
    role: str = "general"
    depth: int = 0
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    status: str = "idle"  # idle, working, completed, failed
    task: str = ""
    result: str = ""
    context: list[dict] = field(default_factory=list)


class AgentOrchestrator:
    """
    Manages a hierarchy of agents that can delegate subtasks.

    Root Agent (user's direct assistant)
      -> Sub-Agent (research)
      -> Sub-Agent (code execution)
        -> Sub-Sub-Agent (testing)

    Each agent maintains its own context and can communicate up/down the chain.
    """

    def __init__(self, brain):
        self.settings = get_settings()
        self.brain = brain
        self.agents: dict[str, AgentNode] = {}
        self.root_agent = self._create_root_agent()
        self.max_depth = self.settings.get("agent.delegation.max_depth", 3)

    def _create_root_agent(self) -> AgentNode:
        agent = AgentNode(
            name="OpenClaw",
            role="primary_assistant",
            depth=0,
        )
        self.agents[agent.id] = agent
        return agent

    async def process_message(
        self,
        message: str,
        session_messages: list[dict] = None,
        memory_context: str = "",
    ) -> dict:
        """Process a user message through the agent hierarchy."""
        self.root_agent.status = "working"
        self.root_agent.task = message

        messages = session_messages or []
        messages.append({"role": "user", "content": message})

        # Check if delegation is needed
        delegation_enabled = self.settings.get("agent.delegation.enabled", True)

        if delegation_enabled:
            # Analyze task complexity
            analysis = await self._analyze_task(message, memory_context)

            if analysis.get("should_delegate") and self.root_agent.depth < self.max_depth:
                return await self._handle_delegation(
                    self.root_agent, message, analysis, messages, memory_context
                )

        # Direct execution
        result = await self.brain.generate(
            messages=messages,
            memory_context=memory_context,
        )

        self.root_agent.status = "completed"
        self.root_agent.result = result.get("content", "")

        return result

    async def _analyze_task(self, message: str, memory_context: str) -> dict:
        """Analyze if a task should be delegated to sub-agents."""
        analysis_prompt = f"""Analyze this task and determine if it should be broken into subtasks.
Respond in JSON format:
{{
    "should_delegate": true/false,
    "complexity": "simple"|"moderate"|"complex",
    "subtasks": ["subtask1", "subtask2"],
    "reasoning": "brief explanation"
}}

Task: {message}"""

        result = await self.brain.generate(
            messages=[{"role": "user", "content": analysis_prompt}],
            memory_context=memory_context,
            max_tokens=500,
        )

        try:
            content = result.get("content", "{}")
            # Extract JSON from response
            import json
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        return {"should_delegate": False, "complexity": "simple", "subtasks": [], "reasoning": ""}

    async def _handle_delegation(
        self,
        parent: AgentNode,
        original_task: str,
        analysis: dict,
        messages: list[dict],
        memory_context: str,
    ) -> dict:
        """Create sub-agents for subtasks and aggregate results."""
        subtasks = analysis.get("subtasks", [])
        if not subtasks:
            return await self.brain.generate(messages=messages, memory_context=memory_context)

        sub_results = []
        for subtask in subtasks:
            child = AgentNode(
                name=f"SubAgent-{len(parent.children_ids)}",
                role="specialist",
                depth=parent.depth + 1,
                parent_id=parent.id,
                task=subtask,
            )
            self.agents[child.id] = child
            parent.children_ids.append(child.id)

            # Execute subtask
            child.status = "working"
            sub_messages = [{"role": "user", "content": subtask}]
            result = await self.brain.generate(
                messages=sub_messages,
                memory_context=memory_context,
                max_tokens=2048,
            )
            child.result = result.get("content", "")
            child.status = "completed"
            sub_results.append({"subtask": subtask, "result": child.result})

        # Aggregate results
        aggregation_prompt = f"""Original task: {original_task}

Subtask results:
{self._format_subtask_results(sub_results)}

Synthesize these results into a complete, coherent response to the original task."""

        messages_for_agg = messages + [{"role": "user", "content": aggregation_prompt}]
        final_result = await self.brain.generate(
            messages=messages_for_agg,
            memory_context=memory_context,
        )

        parent.status = "completed"
        parent.result = final_result.get("content", "")
        return final_result

    def _format_subtask_results(self, results: list[dict]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"### Subtask {i}: {r['subtask']}\n{r['result']}")
        return "\n\n".join(parts)

    def get_hierarchy(self) -> dict:
        """Return the current agent hierarchy as a tree."""

        def build_tree(agent_id: str) -> dict:
            agent = self.agents[agent_id]
            return {
                "id": agent.id,
                "name": agent.name,
                "role": agent.role,
                "status": agent.status,
                "task": agent.task[:100] if agent.task else "",
                "children": [build_tree(cid) for cid in agent.children_ids if cid in self.agents],
            }

        return build_tree(self.root_agent.id)

    def reset(self):
        """Reset the orchestrator for a new conversation."""
        self.agents.clear()
        self.root_agent = self._create_root_agent()
